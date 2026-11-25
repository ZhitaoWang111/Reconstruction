"""Stage 4: the structure handoff for scene assembly, from one finished appearance-agent run.

    PYTHONPATH=src .venv-sam3d/bin/python -m structuregen.export.handoff [agent run dir] [--out <dir>]

Default input is the newest real agent run under ``output/<scene>/structuregen/latest``; it must
have committed. Writes ``output/<scene>/structuregen/structure_handoff/``:

    manifest.json   version, sources with sha256, frame, conventions, the checks below
    structure.json  every segment: prim path, footprint, room-side normal, openings, the colour of
                    each face and every trim, and every floor; read back from structure.usdc,
                    not copied from the run
    structure.usdc  /Structure/Walls/<segment>/<segment>_body + trims as children, and
                    /Structure/Floors/<floor>/<floor>_body; every mesh a static collider
    textures/       tile images of floors (or walls) laid with sg.tiles
    README.md       for people

Steps: Blender exports appearance.blend (blender_export.py), pxr strips Blender bookkeeping,
adds static collision (CollisionAPI, triangle mesh, no RigidBodyAPI) and stage metadata, saves
usdc, then the file is opened again and checked against plan.json: body volumes, every face zone
bound to a material that carries its sRGB, trims parented and inside their segment's length,
colliders present, nothing dynamic. Any failure stops before the handoff directory is replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from .. import config
from ..common.usd_layout import regroup
from ..whitebox.validate import quad_volume

SCRIPT = Path(__file__).with_name("blender_export.py")
SCHEMA = "structure_handoff/v2"
WALLS, FLOORS = "/Structure/Walls", "/Structure/Floors"
VOLUME_TOLERANCE_M3 = 1e-4
INSIDE_SHARE = 0.2                 # an object counts as "in a wall" when this share of its scan points is
FRAMES = ("amb3r_floor", "scannet_scan")
OBJECTS = {"amb3r_floor": config.HANDOFF_ROOT / config.SCENE / "gt_amb3r_floor",     # object scene handoff per frame
           "scannet_scan": config.HANDOFF_ROOT / config.SCENE / "gt"}
ZONES = ("front", "back", "top", "bottom", "end_start", "end_end", "reveal")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def white_model(agent: Path) -> Path:
    """The white-model run an agent run worked on: recorded in its report, else its parent directory."""
    rec = json.loads((agent / "report.json").read_text()).get("white_model")
    return Path(rec) if rec else agent.parent


def latest_agent(run: Path) -> Path:
    runs = sorted(d for d in run.glob("agent_2*") if (d / "report.json").is_file())
    if not runs:
        raise SystemExit(f"no agent run with a report under {run}")
    return runs[-1]


# --------------------------------------------------------------------------------- frame

def frame_transform(frame: str, walls_doc: dict, plan: dict) -> dict | None:
    """amb3r_floor -> ``frame`` as a 4x4 (column vectors), keeping the walls upright.

    The GT handoffs relate the two frames by a rigid matrix (gt_amb3r_floor manifest,
    ``scannet_scan_to_frame``). Applied as is it would tilt the walls by its 0.45 deg of tilt
    and leave their bottoms 2-13 cm above a floor that is flat in scannet_scan. So only its yaw
    is kept; the horizontal shift is fitted to the full matrix over every wall corner (bottom
    and top), and the walls' z = 0 goes to the target handoff's floor_z. The residuals against
    the full matrix are returned with it.
    """
    native = walls_doc["coordinate_frame"]
    if frame == native == "amb3r_floor":
        return None
    if frame == native == "scannet_scan":
        # the walls handoff measures z from the annotated floor; scannet_scan itself does not
        fz = float(walls_doc["frame"]["floor_z"])
        T = np.eye(4)
        T[2, 3] = fz
        return {"matrix": T, "from": "scannet_scan, z from the annotated floor (walls handoff)", "to": frame,
                "native": True, "yaw_deg": 0.0, "translation_m": [0.0, 0.0, fz], "floor_z": fz,
                "source": f"walls handoff frame.floor_z ({walls_doc['frame'].get('floor_z_definition', '')})",
                "target_handoff": str(OBJECTS[frame])}
    if native != "amb3r_floor":
        raise SystemExit(f"no way to move walls from {native} to {frame}")
    v3 = json.loads((OBJECTS["amb3r_floor"] / "manifest.json").read_text())
    target = json.loads((OBJECTS[frame] / "manifest.json").read_text())
    if v3.get("coordinate_frame") != walls_doc["coordinate_frame"] or target.get("coordinate_frame") != frame:
        raise SystemExit(f"cannot relate {walls_doc['coordinate_frame']} to {frame} from the GT handoffs")
    full = np.linalg.inv(np.array(v3["scannet_scan_to_frame"]["matrix"], float))     # amb3r_floor -> scannet_scan
    yaw = float(np.arctan2(full[1, 0], full[0, 0]))
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    corners = np.array([[p[0], p[1], z] for seg in plan["segments"] for p in seg["front"] + seg["back"]
                        for z in (0.0, seg["height_m"])], float)
    exact = corners @ full[:3, :3].T + full[:3, 3]
    shift = (exact - corners @ rz.T)[:, :2].mean(0)
    floor_z = float(target["floor_z"])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rz, [shift[0], shift[1], floor_z]
    mine = corners @ T[:3, :3].T + T[:3, 3]
    horiz = np.linalg.norm((mine - exact)[:, :2], axis=1)
    tilt = float(np.degrees(np.arccos(np.clip(full[2, 2], -1, 1))))
    return {"matrix": T, "from": walls_doc["coordinate_frame"], "to": frame,
            "yaw_deg": round(float(np.degrees(yaw)), 4), "translation_m": T[:3, 3].round(5).tolist(),
            "floor_z": floor_z, "dropped_tilt_deg": round(tilt, 3),
            "source": f"{OBJECTS['amb3r_floor'] / 'manifest.json'} scannet_scan_to_frame (inverted)",
            "source_note": v3["scannet_scan_to_frame"].get("use"),
            "target_handoff": str(OBJECTS[frame]),
            "residual_vs_full_matrix_m": {"horizontal_mean": round(float(horiz.mean()), 4),
                                          "horizontal_max": round(float(horiz.max()), 4),
                                          "wall_bottom_z_full_matrix": [round(float(exact[::2, 2].min()), 4),
                                                                        round(float(exact[::2, 2].max()), 4)]}}


# --------------------------------------------------------------------------------- write

def export_usd(agent: Path, target: Path, frame: str, T: np.ndarray | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "structure.usda"
        r = subprocess.run([str(config.BLENDER), "-b", str(agent / "appearance.blend"), "--python", str(SCRIPT),
                            "--", str(raw)], capture_output=True, text=True)
        if r.returncode or not raw.is_file():
            raise SystemExit(f"Blender export failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        stage = Usd.Stage.Open(str(raw))
        regroup(stage)
        finish_stage(stage, frame, T)
        stage.GetRootLayer().Export(str(target))
        if (Path(tmp) / "textures").is_dir():
            shutil.copytree(Path(tmp) / "textures", target.parent / "textures", dirs_exist_ok=True)


def finish_stage(stage: Usd.Stage, frame: str, T: np.ndarray | None) -> None:
    """Drop Blender bookkeeping, add static collision, move to ``frame``, state it on the stage.

    The frame change is baked into each body's own transform, so /Structure stays the identity
    and every segment Xform still sits at its front start corner in the delivered frame.
    """
    if T is not None:
        move = Gf.Matrix4d(T.T.tolist())               # USD composes row vectors: p @ local @ move
        bodies = list(stage.GetPrimAtPath(WALLS).GetChildren()) + list(stage.GetPrimAtPath(FLOORS).GetChildren())
        for seg in bodies:
            if not seg.IsA(UsdGeom.Xform) or seg.GetName().startswith("_"):
                continue
            xf = UsdGeom.Xformable(seg)
            ops = xf.GetOrderedXformOps()
            if not ops:                                 # floors sit at the origin: no op written
                xf.AddTransformOp().Set(move)
                continue
            if len(ops) != 1 or ops[0].GetOpType() != UsdGeom.XformOp.TypeTransform:
                raise SystemExit(f"{seg.GetPath()}: expected one matrix transform op")
            ops[0].Set(ops[0].Get() * move)
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetName().startswith("userProperties:blender:"):
                prim.RemoveProperty(attr.GetName())
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            # triangle mesh: exact for static bodies; a wall with a doorway or an L-shaped room is not convex
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
    root = stage.GetPrimAtPath("/Structure")
    root.ClearCustomDataByKey("Blender")
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind("assembly")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.GetRootLayer().customLayerData = {"schema": SCHEMA, "frame": frame}
    stage.GetRootLayer().documentation = "Structure handoff: walls and floors with appearance; see manifest.json"


# --------------------------------------------------------------------------------- read back

def faces_of(mesh: UsdGeom.Mesh):
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    idx = list(mesh.GetFaceVertexIndicesAttr().Get())
    faces, k = [], 0
    for c in counts:
        faces.append(idx[k:k + c])
        k += c
    return faces


def material_of(prim) -> dict:
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return {}
    m = mat.GetPrim()
    rgb = m.GetAttribute("userProperties:sg_rgb").Get()
    item = m.GetAttribute("userProperties:sg_item").Get()
    rough = None
    for child in m.GetChildren():
        shader = UsdShade.Shader(child)
        if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface":
            rough = shader.GetInput("roughness").Get()
    return {"material": str(m.GetPath()), "rgb": list(rgb) if rgb is not None else None,
            "item": item, "roughness": None if rough is None else round(float(rough), 3)}


def world_points(prim) -> np.ndarray:
    pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    return (np.c_[pts, np.ones(len(pts))] @ xf)[:, :3]          # USD: row vectors


def basis_of(item, declared_basis) -> str | None:
    """photo: the agent set it from key frames; inferred: from other walls or assumed; manual: a person."""
    if item == "manual":
        return "manual"
    if item == "t1":
        return "photo" if declared_basis else "inferred"
    return "inferred" if item == "w" else None


def read_back(path: Path, plan: dict, report: dict, T: np.ndarray | None) -> tuple[list, list, dict]:
    """structure.json's segments, the failures found, and counts; all from the saved file."""
    stage = Usd.Stage.Open(str(path))
    failures, segments = [], []
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        failures.append("stage is not Z-up metres")
    if str(stage.GetDefaultPrim().GetPath()) != "/Structure":
        failures.append("defaultPrim is not /Structure")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            failures.append(f"{prim.GetPath()}: dynamic API on a static structure")
    planned = {s["id"] for s in plan["segments"]}
    extra = sorted(p.GetName() for p in stage.GetPrimAtPath(WALLS).GetChildren()
                   if p.IsA(UsdGeom.Xform) and p.GetName() not in planned)
    if extra:
        failures.append(f"unexpected prims under {WALLS}: " + ", ".join(extra))
    loose = sorted(p.GetName() for p in stage.GetPrimAtPath("/Structure").GetChildren()
                   if p.GetName() not in ("Walls", "Floors", "_materials"))
    if loose:
        failures.append("prims outside Walls / Floors: " + ", ".join(loose))

    trims_total = 0
    for i, seg in enumerate(plan["segments"]):
        sid = seg["id"]
        xform = stage.GetPrimAtPath(f"{WALLS}/{sid}")
        body = stage.GetPrimAtPath(f"{WALLS}/{sid}/{sid}_body")
        if not body or not body.IsA(UsdGeom.Mesh):
            failures.append(f"{sid}: no body at {WALLS}/{sid}/{sid}_body")
            continue
        mesh = UsdGeom.Mesh(body)
        volume = quad_volume([tuple(p) for p in mesh.GetPointsAttr().Get()], faces_of(mesh))
        if abs(volume - seg["expected_volume_m3"]) > VOLUME_TOLERANCE_M3:
            failures.append(f"{sid}: body volume {volume:.5f} m3, plan {seg['expected_volume_m3']:.5f}")
        if not body.HasAPI(UsdPhysics.CollisionAPI):
            failures.append(f"{sid}: body has no collider")

        declared_basis = ((report["wall_state"].get(sid, {}).get("declared") or {}).get("surface") or {}).get("basis", [])
        faces = {}
        for sub in UsdGeom.Subset.GetAllGeomSubsets(mesh):
            zone = sub.GetPrim().GetName()[len(sid) + 1:]
            m = material_of(sub.GetPrim())
            faces[zone] = {"subset": str(sub.GetPrim().GetPath()), **m,
                           "basis": basis_of(m.get("item"), declared_basis)}
            if not m.get("rgb"):
                failures.append(f"{sid}_{zone}: no material with sg_rgb")
        expect = {"front", "back", "top", "bottom", "end_start", "end_end"} | ({"reveal"} if seg["openings"] else set())
        if set(faces) != expect:
            failures.append(f"{sid}: face zones {sorted(faces)}, expected {sorted(expect)}")

        trims = []
        for child in xform.GetChildren():
            if child == body or not child.IsA(UsdGeom.Xform):
                continue
            kind = child.GetAttribute("userProperties:sg").Get()
            meshes = [p for p in Usd.PrimRange(child) if p.IsA(UsdGeom.Mesh)]
            if kind not in ("skirting", "rail", "casing") or len(meshes) != 1:
                failures.append(f"{child.GetPath()}: not a trim (kind {kind!r}, {len(meshes)} meshes)")
                continue
            m = material_of(meshes[0])
            pts_world = world_points(meshes[0])
            local = np.array(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get(), float)
            # segment frame: x along the wall from its start, y into the room... check the trim stays on its wall
            if local[:, 0].min() < -1e-4 or local[:, 0].max() > seg["length_m"] + 1e-4:
                failures.append(f"{child.GetPath()}: runs past its segment")
            if not meshes[0].HasAPI(UsdPhysics.CollisionAPI):
                failures.append(f"{child.GetPath()}: no collider")
            get = lambda n: child.GetAttribute(f"userProperties:{n}").Get()
            t = {"prim": str(child.GetPath()), "mesh": str(meshes[0].GetPath()), "kind": kind,
                 "rgb": m.get("rgb"), "roughness": m.get("roughness"),
                 "depth_m": round(float(get("sg_depth")), 4)}
            if kind == "casing":
                t.update(opening=int(get("sg_opening")), width_m=round(float(get("sg_width")), 4),
                         sides=str(get("sg_sides")).lower())
            else:
                z, a = get("sg_z"), get("sg_along")
                t.update(z_m=[round(float(z[0]), 4), round(float(z[1]), 4)],
                         along_m=[round(float(a[0]), 4), round(float(a[1]), 4)])
            t["bounds_floor"] = {"minimum": pts_world.min(0).round(4).tolist(), "maximum": pts_world.max(0).round(4).tolist()}
            trims.append(t)
        trims_total += len(trims)

        state = report["wall_state"].get(sid, {})
        declared = state.get("declared") or {}
        move = (lambda q: (T[:2, :2] @ np.asarray(q, float) + T[:2, 3])) if T is not None else (lambda q: np.asarray(q, float))
        front = [move(q) for q in seg["front"]]
        back = [move(q) for q in seg["back"]]
        f0, f1, b0 = front[0], front[1], back[0]
        # the Xform the file holds must agree with the plan's corner, moved
        origin = np.array(UsdGeom.Xformable(xform).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))[3, :3]
        if np.linalg.norm(origin[:2] - f0) > 1e-4:
            failures.append(f"{sid}: Xform origin {origin[:2].round(4)} is not the front start corner {f0.round(4)}")
        u = (f1 - f0) / np.linalg.norm(f1 - f0)
        n_room = np.array([-u[1], u[0]])                        # room on the left of start -> end
        if n_room @ (b0 - f0) > 0:
            n_room = -n_room
        segments.append({
            "number": i + 1, "id": sid, "prim": f"{WALLS}/{sid}", "body": f"{WALLS}/{sid}/{sid}_body",
            "handoff_wall": seg["wall"], "role": seg["role"],
            "along_m": [round(seg["a0"], 4), round(seg["a1"], 4)], "length_m": round(seg["length_m"], 4),
            "thickness_m": seg["thickness_m"], "height_m": seg["height_m"],
            "base_z": round(float(origin[2]), 5),
            "front_floor": [[round(float(c), 5) for c in p] for p in front],
            "back_floor": [[round(float(c), 5) for c in p] for p in back],
            "normal_room": [round(float(n_room[0]), 4), round(float(n_room[1]), 4), 0.0],
            "openings": [{"index": k, "from_m": o["from_m"], "to_m": o["to_m"], "z_floor": o["z_floor"],
                          "kind": "opening", **({"handoff_m": o["handoff_m"]} if "handoff_m" in o else {})}
                         for k, o in enumerate(seg["openings"])],
            "volume_m3": round(volume, 5),
            "faces": dict(sorted(faces.items(), key=lambda kv: ZONES.index(kv[0]))),
            "trims": sorted(trims, key=lambda t: t["prim"]),
            "agent": {"status": state.get("status"), "material": (declared.get("surface") or {}).get("material"),
                      "basis_frames": (declared.get("surface") or {}).get("basis", []),
                      "notes": declared.get("notes"), "summary": state.get("summary")},
        })
    floors, floor_failures = read_floors(stage, plan, report, T)
    failures += floor_failures
    counts = {"segments": len(segments), "floors": len(floors), "trims": trims_total,
              "materials": len(list(p for p in stage.Traverse() if p.IsA(UsdShade.Material))),
              "colliders": len(list(p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)))}
    return segments, floors, failures, counts


def tile_spec(prim) -> dict | None:
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return None
    raw = mat.GetPrim().GetAttribute("userProperties:sg_tiles")
    if not raw or raw.Get() is None:
        return None
    spec = json.loads(raw.Get())
    for child in Usd.PrimRange(mat.GetPrim()):
        sh = UsdShade.Shader(child)
        if sh and sh.GetIdAttr().Get() == "UsdUVTexture":
            spec["texture"] = str(sh.GetInput("file").Get().path)
    return spec


def read_floors(stage, plan, report, T) -> tuple[list, list]:
    """structure.json's floors from the saved file, and what is wrong with them."""
    out, bad = [], []
    t = plan.get("floor_thickness_m", 0.05)
    for f in plan.get("floors", []):
        fid = f["id"]
        body = stage.GetPrimAtPath(f"{FLOORS}/{fid}/{fid}_body")
        if not body or not body.IsA(UsdGeom.Mesh):
            bad.append(f"{fid}: no body at {FLOORS}/{fid}/{fid}_body")
            continue
        mesh = UsdGeom.Mesh(body)
        volume = quad_volume([tuple(p) for p in mesh.GetPointsAttr().Get()], faces_of(mesh))
        if abs(volume - f["area_m2"] * t) > VOLUME_TOLERANCE_M3:
            bad.append(f"{fid}: volume {volume:.5f} m3, area x thickness {f['area_m2'] * t:.5f}")
        if not body.HasAPI(UsdPhysics.CollisionAPI):
            bad.append(f"{fid}: no collider")
        state = report["wall_state"].get(fid, {})
        basis = ((state.get("declared") or {}).get("surface") or {}).get("basis", [])
        faces = {}
        for sub in UsdGeom.Subset.GetAllGeomSubsets(mesh):
            zone = sub.GetPrim().GetName()[len(fid) + 1:]
            m = material_of(sub.GetPrim())
            faces[zone] = {"subset": str(sub.GetPrim().GetPath()), **m, "basis": basis_of(m.get("item"), basis)}
            tiles = tile_spec(sub.GetPrim())
            if tiles:
                faces[zone]["tiles"] = tiles
                tex = Path(stage.GetRootLayer().realPath).parent / tiles.get("texture", "")
                if not tiles.get("texture") or not tex.is_file():
                    bad.append(f"{fid}_{zone}: tile texture {tiles.get('texture')} missing next to the USD")
            if not m.get("rgb"):
                bad.append(f"{fid}_{zone}: no material with sg_rgb")
        if set(faces) != {"top", "side", "bottom"}:
            bad.append(f"{fid}: face zones {sorted(faces)}, expected bottom, side, top")
        pts = (np.c_[np.array(mesh.GetPointsAttr().Get(), float), np.ones(len(mesh.GetPointsAttr().Get()))]
               @ np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())))[:, :3]
        top_z = float(pts[:, 2].max())
        move = (lambda q: (T[:2, :2] @ np.asarray(q, float) + T[:2, 3])) if T is not None else (lambda q: np.asarray(q, float))
        declared = state.get("declared") or {}
        out.append({"id": fid, "kind": f["kind"], "prim": f"{FLOORS}/{fid}", "body": f"{FLOORS}/{fid}/{fid}_body",
                    "area_m2": round(f["area_m2"], 4), "top_z": round(top_z, 5), "thickness_m": t,
                    "top_polygons": [[[round(float(c), 5) for c in move(q)] for q in poly] for poly in f["top_polygons"]],
                    **({"opening_of": f["sg_opening_of"]} if "sg_opening_of" in f else {}),
                    "volume_m3": round(volume, 5),
                    "faces": dict(sorted(faces.items(), key=lambda kv: ("top", "side", "bottom").index(kv[0]))),
                    "agent": {"status": state.get("status"), "material": (declared.get("surface") or {}).get("material"),
                              "basis_frames": basis, "notes": declared.get("notes"), "summary": state.get("summary")}})
    return out, bad


# --------------------------------------------------------------------------------- objects

def objects_in_walls(objects_dir: Path, segments: list, frame: str) -> dict | None:
    """Objects of the scene handoff whose scan points sit inside the wall bodies.

    Measured, not inferred: a point is inside a segment when it lies between its ends, between
    its room-side and back faces, and below its top. Closed doors and windows show up here
    because the walls are solid where they are; so do things hung on walls when the wall face
    and the object scan disagree by a few centimetres.
    """
    if not (objects_dir / "objects.json").is_file():
        return None
    manifest = json.loads((objects_dir / "manifest.json").read_text())
    if manifest.get("coordinate_frame") != frame:
        return None
    rows = []
    for o in json.loads((objects_dir / "objects.json").read_text()):
        if o["label"] in ("wall", "floor", "ceiling"):
            continue
        pts = np.asarray(trimesh.load(objects_dir / o["cloud"]).vertices, float)
        best = None
        for s in segments:
            f0, f1 = np.array(s["front_floor"][0]), np.array(s["front_floor"][1])
            u = (f1 - f0) / np.linalg.norm(f1 - f0)
            n = -np.array(s["normal_room"][:2])
            a, d = (pts[:, :2] - f0) @ u, (pts[:, :2] - f0) @ n
            h = pts[:, 2] - s["base_z"]                                 # height above the wall's base
            inside = (a > 0) & (a < s["length_m"]) & (d > 0) & (d < s["thickness_m"]) & (h < s["height_m"])
            for op in s["openings"]:
                lo, hi = op["from_m"] - s["along_m"][0], op["to_m"] - s["along_m"][0]
                inside &= ~((a > lo) & (a < hi) & (h > op["z_floor"][0]) & (h < op["z_floor"][1]))
            if inside.any() and (best is None or inside.mean() > best[0]):
                best = (inside.mean(), s["id"], a[inside], d[inside], h[inside])
        if best and best[0] >= INSIDE_SHARE:
            share, sid, a, d, z = best
            rows.append({"object_id": o["object_id"], "label": o["label"], "structural": o.get("structural", False),
                         "segment": sid, "points_inside_share": round(float(share), 3),
                         "along_m": [round(float(a.min()), 3), round(float(a.max()), 3)],
                         "z_m": [round(float(z.min()), 3), round(float(z.max()), 3)],
                         "depth_into_wall_p95_m": round(float(np.percentile(d, 95)), 3)})
    rows.sort(key=lambda r: (r["segment"], r["along_m"][0]))
    return {"source": str(objects_dir), "schema_version": manifest.get("schema_version"),
            "objects_sha256": sha256(objects_dir / "objects.json"),
            "rule": f"at least {INSIDE_SHARE:.0%} of the object's scan points inside a wall body "
                    "(between its ends, its two faces, below its top, outside its openings)",
            "along": "segment-local: metres from the segment's start; z_m: height above the wall's base",
            "objects": rows}


# --------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent", type=Path, nargs="?", help="agent run dir; default the newest real one")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--frame", choices=FRAMES, default=None,
                        help="delivered frame; default the walls handoff's own. amb3r_floor walls go to "
                             "scannet_scan with the GT handoffs' matrix (see frame_transform)")
    parser.add_argument("--objects", type=Path, default=None,
                        help="object scene handoff in the delivered frame, for the objects-in-walls check")
    args = parser.parse_args(argv)
    run_root = (config.OUTPUT_ROOT / config.SCENE / "structuregen/latest").resolve()
    agent = (args.agent or latest_agent(run_root)).resolve()
    run = white_model(agent)
    report = json.loads((agent / "report.json").read_text())
    if report["status"] != "ok" or not any(w["wall"] == "wrapup" and w["status"] == "committed" for w in report["walls"]):
        raise SystemExit(f"{agent}: the agent run did not commit (status {report['status']}); nothing to hand off")
    unconverged = [w["wall"] for w in report["walls"] if w["status"] not in ("done", "committed")]
    plan = json.loads((run / "plan.json").read_text())
    handoff = Path(plan["handoff_dir"])
    walls_doc = json.loads((handoff / "walls.json").read_text())
    args.frame = args.frame or walls_doc["coordinate_frame"]
    default_out = "structure_handoff" + ("" if args.frame == "amb3r_floor" else f"_{args.frame}")
    out = (args.out or config.OUTPUT_ROOT / config.SCENE / "structuregen" / default_out).resolve()
    xf = frame_transform(args.frame, walls_doc, plan)
    T = xf["matrix"] if xf else None

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="structure_handoff_", dir=out.parent))
    try:
        usd = staging / "structure.usdc"
        export_usd(agent, usd, args.frame, T)
        segments, floors, failures, counts = read_back(usd, plan, report, T)
        if failures:
            for f in failures:
                print("EXPORT_FAILURE", f, file=sys.stderr)
            raise SystemExit("structure handoff not written: the exported USD failed its checks")

        in_walls = objects_in_walls(args.objects or OBJECTS[args.frame], segments, args.frame)
        structure = {"schema_version": SCHEMA, "scene_id": plan["scene_id"], "coordinate_frame": args.frame,
                     "units": "meters", "up_axis": "Z", "usd": "structure.usdc", "segments": segments,
                     "floors": floors,
                     "objects_in_walls": in_walls}
        (staging / "structure.json").write_text(json.dumps(structure, indent=1, ensure_ascii=False))
        storey = walls_doc["storey"]
        manifest = {
            "schema_version": SCHEMA,
            "scene_id": plan["scene_id"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": {"usd": "structure.usdc", "structure": "structure.json",
                      "textures": sorted(str(x.relative_to(staging)) for x in (staging / "textures").glob("*"))
                      if (staging / "textures").is_dir() else [],
                      "usd_sha256": sha256(usd), "structure_sha256": sha256(staging / "structure.json")},
            "source_handoff": {"path": str(handoff), "schema_version": walls_doc["schema_version"],
                               "walls_sha256": sha256(handoff / "walls.json"),
                               "producer_run": walls_doc["producer"].get("run")},
            "amendments": json.loads((agent / "amendments.json").read_text()) if (agent / "amendments.json").is_file() else [],
            "source_run": {"white_model": str(run), "plan_sha256": sha256(run / "plan.json"),
                           "agent": str(agent), "appearance_blend_sha256": sha256(agent / "appearance.blend"),
                           "agent_model": report["model"], "agent_replies": report["replies"],
                           "agent_usd": report["usd"], "unconverged_walls": unconverged},
            "coordinate_frame": args.frame,
            "frame": frame_doc(args.frame, walls_doc, xf),
            "units": "meters", "up_axis": "Z",
            "conventions": {
                "prims": "/Structure (defaultPrim, kind assembly) / Walls / <segment> (Xform at the segment's front "
                         "start corner: +X along the wall, +Y into the room, +Z up) / <segment>_body (Mesh) and "
                         "<segment>__<kind>__<item> (trim Xform with one Mesh); /Structure / Floors / <floor> / "
                         "<floor>_body (Mesh, world coordinates). Materials under /Structure/_materials",
                "floors": "one floor per room (the inner outline minus every wall footprint, room_1 the largest) and "
                          "a threshold under each floor-level opening; top face at the walls' base, slab "
                          "floor_thickness_m below it. Faces: top, side, bottom",
                "tiles": "a face laid with tiles carries UV primvar st (1 unit = one tile) and a UsdUVTexture "
                         "(textures/<face>_tiles.png, wrap repeat) feeding diffuseColor; userProperties:sg_tiles holds "
                         "size_m, joint_m, joint_rgb and angle_deg; sg_rgb is the tile colour",
                "faces": "each body face belongs to a zone, a GeomSubset <body>_<zone> bound to its own material. "
                         "Walls: front (room side), back, top, bottom, end_start, end_end, reveal (opening jambs and "
                         "head). Floors: top, side, bottom",
                "colour": "materials are UsdPreviewSurface; inputs:diffuseColor is linear, userProperties:sg_rgb "
                          "is the same colour in sRGB 0-255 as the agent chose it",
                "basis": "faces[].basis: photo = the agent set it from the wall's key frames; inferred = the agent "
                         "inferred it from other walls (a wall with no key frame, or the wrap-up's hidden faces); "
                         "manual = changed by hand after the run, see amendments",
                "physics": "every Mesh (walls, trims, floors) has CollisionAPI + MeshCollisionAPI approximation none "
                           "(triangle mesh); no RigidBodyAPI, so they are static colliders. No PhysicsScene is "
                           "authored; the assembled scene owns it",
                "openings": "only the walls handoff's openings are holes, narrowed by plan.opening_inset_m per side "
                            "(handoff range kept as handoff_m); closed doors and windows "
                            "are solid wall. Door leaves, windows and window casings come from instance generation",
                "storey": f"height {storey['height_m']:.3f} m is {storey['kind']} (walls handoff storey); no ceiling",
            },
            "checks": {"passed": True, **counts,
                       "volumes": "each wall equals footprint x height minus openings, each floor top area x "
                                  "thickness (plan.json), within 1e-4 m3",
                       "zones": "every face zone present and bound to a material with sg_rgb",
                       "trims": "every trim is a child of its segment and stays within its length",
                       "static": "no RigidBodyAPI / MassAPI anywhere",
                       "objects_in_walls": None if in_walls is None else len(in_walls["objects"])},
            "storey": {"height_m": walls_doc["storey"]["height_m"], "kind": walls_doc["storey"]["kind"]},
            "floor_thickness_m": plan.get("floor_thickness_m"),
            "not_included": ["ceiling", "door_leaves", "windows", "window_casings", "furniture"],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        (staging / "README.md").write_text(readme(manifest, structure))
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"STRUCTURE_HANDOFF {out}: {counts['segments']} segments, {counts['floors']} floors, {counts['trims']} trims, "
          f"{counts['materials']} materials, {counts['colliders']} colliders")
    return 0


def frame_doc(frame: str, walls_doc: dict, xf: dict | None) -> dict:
    native = {"definition": walls_doc["frame"]["definition"], "file": walls_doc["frame"].get("file"),
              "scale": walls_doc["frame"].get("scale")}
    if xf is not None and xf.get("native"):
        return {"definition": walls_doc["frame"].get("definition"), "floor_z": xf["floor_z"],
                "floor": f"wall bases at z = {xf['floor_z']} ({walls_doc['frame'].get('floor_z_definition', '')})",
                "same_as": f"the object scene handoff {xf['target_handoff']} and assembly_handoff/v2",
                "from_walls_handoff": {k: (v.round(9).tolist() if k == "matrix" else v) for k, v in xf.items()}}
    if xf is None:
        return {**native, "floor": "z = 0 is the floor AMB3R reconstructed (the walls stand on it)",
                "same_as": "walls/v2 and the object scene handoff handoff/v3 (gt_amb3r_floor); "
                           "not scannet_scan, the frame of the gt/ handoff and of assembly_handoff/v2"}
    return {"same_as": f"the object scene handoff {xf['target_handoff']} and assembly_handoff/v2",
            "floor": f"wall bases at z = {xf['floor_z']} (that handoff's floor_z)",
            "from_native": {**{k: v for k, v in xf.items() if k != "matrix"},
                            "matrix_column_vectors": xf["matrix"].round(9).tolist(),
                            "native": {"frame": xf["from"], **native}}}


def readme(manifest: dict, structure: dict) -> str:
    rows = []
    for s in structure["segments"]:
        f = s["faces"]["front"]
        trims = "；".join(
            (f"门套 面宽 {t['width_m'] * 100:.1f} cm（{'仅房间一侧' if t['sides'] == 'front' else t['sides']}）"
             if t["kind"] == "casing" else
             f"{'踢脚线' if t['kind'] == 'skirting' else '腰线'} z {t['z_m'][0]:.2f}–{t['z_m'][1]:.2f} m")
            + f" rgb {tuple(t['rgb'])}" for t in s["trims"]) or "—"
        op = "；".join(f"{o['from_m']:.3f}–{o['to_m']:.3f} m × {o['z_floor'][0]}–{o['z_floor'][1]} m"
                       for o in s["openings"]) or "—"
        rows.append(f"| {s['number']} | `{s['id']}` | {'外墙' if s['role'] == 'exterior' else '内墙'} | "
                    f"{s['length_m']:.3f} | {s['thickness_m']} | {tuple(f['rgb'])} | {trims} | {op} |")
    src = manifest["source_run"]
    holes = "；".join(f"`{s['id']}` 沿墙 {o['from_m']:.2f}–{o['to_m']:.2f} m、高 {o['z_floor'][0]}–{o['z_floor'][1]} m"
                     for s in structure["segments"] for o in s["openings"]) or "没有"
    fr = manifest["frame"]
    amend = "".join(f"\n\n**人工修改**（{a['at']}）：{a['reason']}\n" + "\n".join(
        f"- `{p['seg']}_{p['face']}`：{tuple(p['was']['rgb'])} → {tuple(p['rgb'])}" for p in a["paint"])
        for a in manifest.get("amendments", []))
    st = manifest["storey"]
    storey = f"{st['height_m']:.2f} m（上游 `{st['kind']}`）"
    casings = "\n".join(f"- `{s['id']}` 的门套只做在{'房间一侧' if t['sides'] == 'front' else '背面' if t['sides'] == 'back' else '两侧'}（agent 的选择）。"
                         for s in structure["segments"] for t in s["trims"] if t["kind"] == "casing" and t["sides"] != "both")
    inset = [o for s in structure["segments"] for o in s["openings"] if o.get("handoff_m")]
    if inset:
        casings += "".join(f"\n- 洞口比上游交接窄：上游给的是标注门框的外沿（含门套），这里每侧收进 "
                           f"{(o['from_m'] - o['handoff_m'][0]):.2f} m，从 {o['handoff_m'][0]:.3f}–{o['handoff_m'][1]:.3f} 收到 "
                           f"{o['from_m']:.3f}–{o['to_m']:.3f} m。" for o in inset)
    if manifest["coordinate_frame"] == "amb3r_floor":
        coords = ("- **`amb3r_floor`**：米，+Z 向上，地面 z = 0，和上游墙体交接 `walls/v2`、物体场景交接 `handoff/v3`（`gt_amb3r_floor`）相同，可以直接叠放。\n"
                  "- **不是** `scannet_scan`：资产交接 `assembly_handoff/v2` 关联的 `gt/` 是 `scannet_scan`，和这份墙直接叠放会错位。"
                  "需要 `scannet_scan` 的版本见 `structure_handoff_scannet_scan/`。\n"
                  "- `handoff/v3` 的 `floor_z = -0.026`：GT 物体站的地面比墙所站的 AMB3R 地面低约 2.6 cm。\n"
                  "- 每段的 Xform 原点在 front 面起点的地面处：+X 沿墙，+Y 指向房间，+Z 向上。`structure.json` 里的坐标都已在世界坐标。")
    elif "from_walls_handoff" in fr:
        x = fr["from_walls_handoff"]
        coords = (f"- **`{manifest['coordinate_frame']}`**：ScanNet 原始扫描坐标系（未套 axisAlignment），米，+Z 向上。和物体场景交接 "
                  f"`{x['target_handoff']}`、资产交接 `assembly_handoff/v2` 相同，可以直接叠放。\n"
                  f"- 墙就是在这个坐标系里识别和生成的（上游墙体交接 `{manifest['source_handoff']['path'].split('/')[-1]}`），没有做旋转平移。"
                  f"唯一的处理：上游墙体交接的 z 从标注地面量起，导出时加回 `floor_z = {x['floor_z']}`，所以墙底在 z = {x['floor_z']}，"
                  "和 `gt/` 的 `floor_z` 一致。\n"
                  "- 每段的 Xform 原点在 front 面起点的墙底处：+X 沿墙，+Y 指向房间，+Z 向上。`structure.json` 里的坐标都已在这个坐标系；"
                  "`base_z` 是墙底高度，洞口和饰线的 `z` 都从墙底量起。")
    else:
        x = fr["from_native"]
        r = x["residual_vs_full_matrix_m"]
        coords = (f"- **`{manifest['coordinate_frame']}`**：米，+Z 向上，和物体场景交接 `{x['target_handoff']}`、资产交接 `assembly_handoff/v2` 相同，可以直接叠放。\n"
                  f"- 墙是在 `{x['from']}` 里生成的，导出时整体搬过来：绕 Z 转 {x['yaw_deg']:.2f}°，平移 {tuple(x['translation_m'])} m。"
                  f"矩阵（列向量）在 `manifest.json` → `frame.from_native.matrix_column_vectors`。\n"
                  f"- 来源是 GT 交接给的两坐标系刚体变换（`{x['source'].split('/')[-2]}/manifest.json` 的 `scannet_scan_to_frame` 取逆）。"
                  f"上游注明它只用于评估：只有带 GT 的 ScanNet 场景才有。\n"
                  f"- **只取了它的偏航和水平平移，丢掉了 {x['dropped_tilt_deg']}° 的倾斜**：照搬会让墙倾斜、墙底高度在 "
                  f"{r['wall_bottom_z_full_matrix'][0]}–{r['wall_bottom_z_full_matrix'][1]} m 之间起伏。现在墙保持竖直，墙底放在 "
                  f"`floor_z = {x['floor_z']}`（该交接的地面高度）。和完整矩阵相比，墙角水平位置平均差 "
                  f"{r['horizontal_mean'] * 1000:.0f} mm、最大 {r['horizontal_max'] * 1000:.0f} mm。\n"
                  "- 每段的 Xform 原点在 front 面起点的墙底处：+X 沿墙，+Y 指向房间，+Z 向上。`structure.json` 里的坐标都已在这个坐标系；"
                  "`base_z` 是墙底高度，洞口和饰线的 `z` 都从墙底量起。")
    iw = structure.get("objects_in_walls")
    why_hung = ("其余贴墙物体是因为墙面来自 AMB3R 重建，和 GT 物体差几厘米（上游 `known_issues`：房间每边比 GT 小 3–16 cm）"
                if manifest["source_handoff"].get("producer_run") != "gt" else
                "其余贴墙物体是因为墙面取的是标注墙点的平面，挂在墙上的东西（窗帘、镜子、淋浴间）的点有一部分在这个平面后面")
    if iw is None:
        in_walls = "- 没有同坐标系的物体交接，未检查物体与墙的重叠。"
    else:
        lines = [f"| `{r['object_id']}` | {r['label']} | `{r['segment']}` | {r['along_m'][0]:.2f}–{r['along_m'][1]:.2f} | "
                 f"{r['z_m'][0]:.2f}–{r['z_m'][1]:.2f} | {r['points_inside_share']:.0%} | {r['depth_into_wall_p95_m'] * 100:.0f} cm |"
                 for r in iw["objects"]]
        in_walls = ("- 物体扫描点落在墙体里的（`structure.json` → `objects_in_walls`，按 `" + iw["source"].split("/")[-1] + "`" +
                    " 实测，阈值为点数的 " + f"{INSIDE_SHARE:.0%}" + "）。门、窗、门框是因为墙在那里是实的（门框还因为洞口收窄过）；" + why_hung + "：\n\n"
                    "  | 物体 | label | 墙段 | 沿墙 m | 高 m | 点在墙内 | 嵌入深度 p95 |\n  |---|---|---|---|---|---|---|\n"
                    + "\n".join("  " + l for l in lines))
    frows = []
    for f in structure.get("floors", []):
        top = f["faces"]["top"]
        tl = top.get("tiles")
        finish = (f"瓷砖/板 {tl['size_m'][0]:.2f}×{tl['size_m'][1]:.2f} m，缝 {tl['joint_m'] * 1000:.0f} mm "
                  f"{tuple(tl['joint_rgb'])}，转 {tl['angle_deg']:.1f}°" if tl else "纯色")
        kind = "房间" if f["kind"] == "room" else f"门槛（`{f.get('opening_of')}` 洞口下）"
        frows.append(f"| `{f['id']}` | {kind} | {f['area_m2']:.2f} | {tuple(top['rgb'])} | {finish} | {top['basis']} |")
    floor_table = ("\n".join(["| 地板 | 类型 | 面积 m² | 颜色（sRGB） | 铺面 | 来源 |", "|---|---|---|---|---|---|"] + frows)
                   if frows else "（这次没有地板）")
    ft = manifest.get("floor_thickness_m")
    return f"""# 结构体交接文件（{manifest['schema_version']}）

场景 `{manifest['scene_id']}`。交付整间房的墙体和地板：每段墙、每块地板各自独立，带外观（每个面一种颜色，地板可铺瓷砖纹理）
和饰线（踢脚线、腰线、门套），都是静态碰撞体。由 ObjectGeneration 的结构体生成（`src/structuregen`）产出，
下游场景组装直接加载 `structure.usdc`。

## 文件

| 文件 | 内容 |
|---|---|
| `structure.usdc` | 结构体 USD：`/Structure/Walls/<段>/<段>_body` 是墙身，饰线是同一段下的子物体 `<段>__<种类>__<编号>`；`/Structure/Floors/<地板>/<地板>_body` 是地板 |
| `textures/` | 铺了瓷砖的面用的贴图（一块砖加半条缝，按 UV 重复） |
| `structure.json` | 每段墙、每块地板的数据：prim 路径、占地、法向、洞口、每个面的颜色与来源、瓷砖参数、饰线尺寸。**从 structure.usdc 读回生成**，和文件一致 |
| `manifest.json` | 版本、上游与运行的 sha256、坐标系、约定、导出时的校验结果 |

## 坐标

{coords}

## 墙段

| # | 段 | 类型 | 长 m | 厚 m | 房间一侧颜色（sRGB） | 饰线 | 洞口 |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

所有段高 {structure['segments'][0]['height_m']:.3f} m。洞口沿墙位置从所属墙（`handoff_wall`）的起点量起。

## 地板

{floor_table}

地板由墙围出来：外墙内轮廓减去所有墙的占地，剩下的每一块是一个房间；地板顶面和墙底齐平，往下 {ft} m 厚。
落地的洞口下面（墙被切开的地方）另有一块门槛地板，填上墙占地留下的缺口。

## 外观

- 材质是 `UsdPreviewSurface`，`inputs:diffuseColor` 是线性值；`userProperties:sg_rgb` 是同一颜色的 sRGB（0–255）。
- 每个面一个 GeomSubset、一个材质。墙：front（房间一侧）、back、top、bottom、end_start、end_end、reveal（洞口侧壁）。地板：top、side、bottom。
- `faces[].basis`：`photo` 是 agent 看关键帧定的；`inferred` 是推定的（没有关键帧的墙段，或收尾时补的照片看不到的面）；`manual` 是运行后人工改的，见下方「人工修改」和 `manifest.json` → `amendments`。{amend}
- 墙是纯色。地板由 agent 选：纯色，或瓷砖 / 木地板（`sg.tiles`）。铺砖的面带 UV（`primvars:st`，1 个单位 = 1 块砖）和一张重复贴图，
  `userProperties:sg_tiles` 记下砖尺寸、缝宽、缝色和转角；`sg_rgb` 是砖本身的颜色。

## 物理

- 每个 Mesh（墙身、饰线、地板）都带 `CollisionAPI` + `MeshCollisionAPI`（`approximation = none`，三角网格），没有 `RigidBodyAPI`，即静态碰撞体。
- 没有写 `PhysicsScene`，由组装后的场景提供。

## 不包含 / 需要注意

- 不含：天花板、门扇、窗、窗套、家具。
- **墙上只有交接测到的洞**：{holes}。关着的门和窗处墙是实的，下游放门、窗时会和墙相交，需要在墙上开洞或接受相交。
{in_walls}
- 层高 {storey}；墙厚是上游**假定**的，不是测量值。
{casings}

## 来源

- 白模：`{src['white_model']}`
- 外观 agent：`{src['agent']}`（{src['agent_model']}，{src['agent_replies']} 次回复，${src['agent_usd']:.2f}）
- 上游墙体交接：`{manifest['source_handoff']['path']}`（{manifest['source_handoff']['schema_version']}，sha256 `{manifest['source_handoff']['walls_sha256'][:12]}…`）
- 重新生成：`scripts/run_structure_export.sh [agent 运行目录] --frame {manifest['coordinate_frame']}`
"""


if __name__ == "__main__":
    sys.exit(main())
