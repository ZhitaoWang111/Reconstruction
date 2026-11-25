"""Build the assembly handoff: one simulation-ready USD per generated object.

For each successfully generated object the script runs:
  1. mesh_to_usd.py           – GLB → Z-up, scaled USD with estimated appearance
  2. configure_usd_physics.py – adds rigid body, mass, collision to the USD
For an object whose asset came from DualCraft (``generator: dualcraft``) it runs
configure_articulated_physics.py instead, which writes both USDs from the URDF
with one rigid body per moving part and the joints between them.
and records what generation produced: status, source frame, SAM3D pose re-expressed
in the USD asset frame, asset dimensions and how far the size can be trusted.

The scene handoff (handoff/v1) is not copied: labels, clouds, key frames, camera
and placement hints stay there.  Scene assembly reads both handoffs and joins
them on ``object_id``; ``source_handoff`` pins the exact upstream bytes.

Usage::

    PYTHONPATH=src .venv-sam3d/bin/python -m objgen.handoff.build_assembly_handoff output/scene0000_00/gt
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

from ..config import ROOT
from ..to_usd.mesh_ops import decimate, glb_to_usd_points  # noqa: E402

VENV_PYTHON = ROOT / ".venv-sam3d" / "bin" / "python"
MESH_TO_USD = ROOT / "scripts" / "pipeline" / "mesh_to_usd.py"
CONFIGURE_PHYSICS = ROOT / "scripts" / "pipeline" / "configure_usd_physics.py"
CONFIGURE_ARTICULATED = ROOT / "scripts" / "pipeline" / "configure_articulated_physics.py"
TARGET_FACES = 40000  # mesh_to_usd.py default; the pose offset is replayed with it

DEFAULT_PRIM_PREFIX = "Obj"

# PyTorch3D camera (+X left, +Y up, +Z forward) -> OpenCV camera (+X right, +Y down).
P3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])

POSE_CONVENTION = {
    "estimator": "SAM3D layout head, single image (the generation frame)",
    "frame": "generation key frame camera, OpenCV axes (+X right, +Y down, +Z forward)",
    "maps": "USD asset-local point (metres, the coordinates stored in the USD) "
            "-> generation camera point",
    "formula": "p_cam = scale * rotation @ p_usd + translation",
    "to_scene": "p_scene = C2W @ [p_cam, 1], C2W = camera_to_world of the "
                "generation frame in handoff/v1",
    "rotation": "3x3 row-major; usable",
    "scale": "camera units per metre; SAM3D depth is up to an unknown scale, so "
             "scale and translation are not metric (median scan/SAM3D depth "
             "ratio 0.96, IQR 0.86-1.15 on scene0000_00)",
    "derivation": "SAM3D returns q (wxyz), t, s for its Z-up object frame in the "
                  "PyTorch3D camera: x_cam = s * R(q)^T x_obj + t.  The USD asset is "
                  "x_obj scaled by scale_factor and shifted by usd_offset_m, so "
                  "rotation = diag(-1,-1,1) R(q)^T, scale = s / scale_factor, "
                  "translation = diag(-1,-1,1) (t + scale * R(q)^T usd_offset_m).",
    "check": "projecting the USD mesh with this pose into the generation frame gives "
             "silhouette IoU against the handoff mask of median 0.64 (42/52 above 0.5) "
             "on scene0000_00",
    "caution": "the rotation is a single-view guess: on scene0000_00, 6 of 52 are "
               "upside down (about 170-180 deg from upright).  Yaw is unverified: on "
               "15 of 44 upright objects it is 180 deg from the best scan registration, "
               "and geometry alone cannot tell which of the two is right.",
}

SCENE_POSE_CONVENTION = {
    "applies_to": "objects with pose.frame == 'scene' (DualCraft assets)",
    "maps": "USD asset-local point (metres) -> scan point (metres)",
    "formula": "p_scene = matrix @ [p_usd, 1]; matrix is 4x4 row-major, rotation part orthonormal, scale 1",
    "source": "the DualCraft object frame recorded when the handoff was converted to DualCraft inputs: "
              "footprint box fitted to the scan cloud (position, yaw, bottom z) and the front taken as "
              "the box side facing the key-frame cameras",
    "metric": "yes; position and yaw come from the scan, unlike the SAM3D camera-frame pose",
    "caution": "the front/back choice is the camera-side rule, not geometry: a box-like object fits "
               "the scan equally well turned 180 degrees",
}

ARTICULATION_CONVENTION = {
    "applies_to": "objects with articulation != null (DualCraft assets)",
    "bodies": "one rigid body per moving part; links joined by fixed joints in the URDF (handles, "
              "knobs, shelves) are merged into the body they hang from",
    "joints": "PhysX joints under <prim>/joints; limits in radians (revolute) or metres (prismatic); "
              "axis is the URDF axis in the child body frame",
    "fixed_base": "true when body_type is static: <prim>/joints/fix_base pins the root body to the "
                  "world, the doors still move",
    "rest": "the USD is authored with every joint at 0 (doors closed)",
}

SIZE_CONVENTION = {
    "dimensions_m": "asset bounding box in its own frame, X, Y, Z (Z is height)",
    "confidence": "high = scale from scan Z height and the asset agrees with the scan "
                  "on every axis; medium = no contradiction but something is unverified "
                  "(see flags); low = Z height unusable or the asset is under half the "
                  "scanned extent on some axis",
    "axis_ratio": "asset extent / scanned extent for height and the two footprint "
                  "axes (each sorted long to short, so orientation does not matter); "
                  "the scan covers only what was seen, so ratios above 1 are expected "
                  "for partial scans",
}


def _run(cmd: list, label: str) -> None:
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode})\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prim_name(oid: str) -> str:
    """Convert obj_0002 → Obj2 (valid USD identifier, no leading digit)."""
    num = oid.replace("obj_", "").lstrip("0") or "0"
    return DEFAULT_PRIM_PREFIX + num


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def usd_offset(glb: Path, scale: float, conversion: Path) -> list[float]:
    """Shift mesh_to_usd applied after rotating and scaling (XY centre, bottom).

    Read from the conversion report when it has one; older reports predate the
    field, so replay the same decimation and arithmetic instead.
    """
    if conversion.exists():
        recorded = json.loads(conversion.read_text()).get("glb_to_usd")
        if recorded and abs(recorded["scale"] - scale) < 1e-9:
            return recorded["offset_m"]
    mesh = decimate(trimesh.load(glb, force="mesh", process=False), TARGET_FACES)
    return glb_to_usd_points(mesh.vertices, scale)[1].tolist()


def pose_in_usd_frame(pose: dict, scale_factor: float, offset: list[float]) -> dict:
    s = float(pose["scale"][0][0])
    r_obj = quat_to_matrix(np.asarray(pose["rotation"][0], dtype=np.float64)).T
    scale = s / scale_factor
    rotation = P3D_TO_OPENCV @ r_obj
    translation = P3D_TO_OPENCV @ (np.asarray(pose["translation"][0], dtype=np.float64)
                                   + scale * r_obj @ np.asarray(offset))
    return {
        "rotation": np.round(rotation, 6).tolist(),
        "translation": np.round(translation, 6).tolist(),
        "scale": round(scale, 6),
        "usd_offset_m": [round(v, 6) for v in offset],
        "raw": {"rotation_wxyz": pose["rotation"][0], "translation": pose["translation"][0],
                "scale": s, "frame": pose.get("frame")},
    }


def size_block(sz: dict, dimensions: list[float]) -> dict:
    return {
        "dimensions_m": [round(v, 4) for v in dimensions],
        "height_m": sz.get("height_m"),
        "scale_factor": sz.get("scale_factor"),
        "source": sz.get("source"),
        "confidence": sz.get("confidence"),
        "flags": sz.get("flags", []),
        "axis_ratio": sz.get("axis_ratio"),
        "candidates_m": sz.get("candidates_m"),
    }


def physics_block(pa: dict, body_type: str, demotion: str | None) -> dict:
    mass = pa.get("mass_kg", 10.0)
    return {
        "body_type": body_type,
        "mass_kg": mass if body_type in ("dynamic", "kinematic") else None,
        "roughness": pa.get("roughness", 0.45),
        "metallic": pa.get("metallic", 0.0),
        "collision_approximation": pa.get("collision_approximation", "convexHull"),
        "confidence": pa.get("confidence"),
        "estimated_body_type": pa.get("body_type", "dynamic"),
        "demotion_reason": demotion,
        "reasoning": pa.get("reasoning"),
    }


def generation_source(record: dict, generation: Path, handoff: Path) -> dict:
    """Which frame the asset was generated from, and whether the handoff still has those bytes."""
    source = {"view": record.get("view"), **(record.get("source") or {})}
    report_path = generation / record["report"] if record.get("report") else None
    if not (report_path and report_path.exists() and source.get("rgb")):
        return source
    inputs = json.loads(report_path.read_text()).get("inputs", {})
    source["rgb_sha256"] = inputs.get("rgb_sha256")
    source["mask_sha256"] = inputs.get("mask_sha256")
    source["inputs_match_handoff"] = all(
        (handoff / source[k]).exists() and sha256(handoff / source[k]) == source[f"{k}_sha256"]
        for k in ("rgb", "mask"))
    return source


def scene_pose(frame: dict, offset: list[float]) -> dict:
    """USD asset -> scan, from the object frame recorded at conversion.  The frame's
    axes are stored rounded, so they are re-orthonormalised (front kept, up from Z)."""
    front = np.asarray(frame["front"], dtype=np.float64)
    up = np.asarray(frame.get("up", [0, 0, 1]), dtype=np.float64)
    front -= up * front.dot(up) / up.dot(up)
    front /= np.linalg.norm(front)
    up /= np.linalg.norm(up)
    rotation = np.column_stack([front, np.cross(up, front), up])
    origin = np.array([*frame["origin_xy"], frame["bottom_z"]], dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = origin + rotation @ np.asarray(offset, dtype=np.float64)
    return {"frame": "scene", "matrix": np.round(matrix, 6).tolist(),
            "rotation": np.round(rotation, 6).tolist(), "translation": np.round(matrix[:3, 3], 6).tolist(),
            "scale": 1.0, "dualcraft_frame": frame}


def dualcraft_generation(record: dict, stage: Path) -> dict:
    """Where the DualCraft asset came from, and which key frames it was given."""
    conversion = json.loads((stage / "inputs" / "scannet_handoff.json").read_text())
    used = [{"seq": v.get("keyframe"), "file": v.get("file"), "azimuth_deg": v.get("azimuth_deg"),
             "elevation_deg": v.get("elevation_deg")} for v in conversion.get("views", [])]
    return {"generator": "dualcraft", "view": record.get("view"),
            # The front key frame; scene assembly looks frames up by seq.
            "seq": conversion.get("front_frame") or (used[0]["seq"] if used else None),
            "frames": used, **(record.get("source") or {})}


def build_dualcraft_object(record: dict, generation: Path, output: Path, pa: dict, sz: dict, *,
                           dry_run: bool, skip_existing: bool) -> dict:
    oid = record["object_id"]
    label = record.get("label") or ""
    stage = (generation / record["product"]).parent
    usd_visual = output / "usds" / oid / "visual.usdc"
    usd_physics = output / "usds" / oid / "physics.usdc"
    estimate_source = "dualcraft" if pa.get("generator") == "dualcraft" else "stale"
    body_type = pa.get("body_type", "dynamic")
    if body_type not in ("static", "dynamic"):
        # A SAM3D-era estimate may say kinematic for an object with doors; for an
        # articulated asset the doors are joints and the question is only the root.
        body_type = "static"
    mass = float(pa.get("mass_kg", 10.0))
    print(f"{oid}  {label:<22}  mass={mass:.1f}kg  type={body_type:<9}  articulated (dualcraft)"
          + ("  [estimate made for the SAM3D asset; rerun estimate_physics_appearance]"
             if estimate_source == "stale" else ""), flush=True)
    trajectory = stage / record_trajectory(stage)
    if not dry_run and not (skip_existing and usd_physics.exists()):
        usd_physics.parent.mkdir(parents=True, exist_ok=True)
        args = [VENV_PYTHON, CONFIGURE_ARTICULATED, stage / "product", "--output", usd_physics,
                "--visual-output", usd_visual, "--mass", str(mass), "--body-type", body_type,
                "--blueprint-run", stage, "--prim-name", prim_name(oid)]
        if trajectory.is_file():
            args += ["--trajectory", trajectory]
        _run(args, f"configure_articulated_physics {oid}")

    entry = {"object_id": oid, "status": "success", "generator": "dualcraft",
             "generation": dualcraft_generation(record, stage)}
    if dry_run:
        return entry
    report = json.loads(usd_physics.with_suffix(".physics.json").read_text())
    entry["usd"] = f"usds/{oid}/physics.usdc"
    size = size_block(sz, report["dimensions_m"])
    size.update(given_size_m=sz.get("given_size_m"), given_ratio=sz.get("given_ratio"))
    entry["size"] = size
    frame = json.loads((stage / "inputs" / "scannet_handoff.json").read_text())["object_frame"]
    entry["pose"] = scene_pose(frame, report["asset_offset_m"])
    appearance = ({"roughness": pa.get("roughness"), "metallic": pa.get("metallic")}
                  if estimate_source == "dualcraft" else asset_appearance(stage / "product"))
    entry["physics"] = {
        "body_type": report["body_type"], "mass_kg": report["mass_kg"],
        "roughness": appearance["roughness"], "metallic": appearance["metallic"],
        "appearance_source": "asset materials (MTL), per material in the USD",
        "collision_approximation": report["collision_approximation"],
        "collision_hulls": report["collision_hulls"],
        "confidence": "low" if report["density_flags"] else pa.get("confidence"),
        "estimate_source": estimate_source,
        "estimated_body_type": pa.get("body_type"),
        "mass_reference": pa.get("mass_reference"),
        "implied_density_kg_m3": report["implied_density_kg_m3"],
        "density_flags": report["density_flags"],
        "demotion_reason": None,
        "reasoning": pa.get("reasoning"),
    }
    entry["articulation"] = {
        "fixed_base": report["fixed_base"],
        "bodies": [{"name": name, "prim": body["prim"].split("/", 2)[-1], "links": body["links"],
                    "mass_kg": body["mass_kg"]} for name, body in report["bodies"].items()],
        "joints": [{k: j[k] for k in ("name", "type", "parent", "child", "axis", "lower", "upper",
                                      "axis_source", "axis_changed_by_agent")} for j in report["joints"]],
        "checks": report["interference"],
    }
    return entry


def asset_appearance(product: Path) -> dict:
    """Roughness / metallic of a DualCraft asset when no estimate was made for it:
    the same face-weighted mean estimate_physics_appearance records."""
    from ..dualcraft import dualcraft_product as dp
    from ..planning.estimate_physics_appearance import material_appearance
    return material_appearance(dp.read_product(product))


def record_trajectory(stage: Path) -> str:
    report = json.loads((stage / "report.json").read_text())
    return report.get("trajectory") or "trajectory.jsonl"


def build_object(record: dict, generation: Path, handoff: Path, output: Path, pa: dict, sz: dict, *,
                 dry_run: bool, skip_existing: bool) -> dict:
    oid = record["object_id"]
    label = record.get("label") or ""
    scale = sz.get("scale_factor", 1.0)
    body_type = pa.get("body_type", "dynamic")
    coll_approx = pa.get("collision_approximation", "convexHull")

    # Non-watertight meshes can't yield valid inertia.  Demote dynamic → static
    # rather than crashing; the object will still collide correctly.
    demotion = None
    if not record.get("stats", {}).get("watertight", True) and body_type == "dynamic":
        body_type, demotion = "static", "non-watertight mesh"

    glb = generation / record["mesh"]
    usd_visual = output / "usds" / oid / "visual.usdc"
    usd_physics = output / "usds" / oid / "physics.usdc"

    print(f"{oid}  {label:<22}  mass={pa.get('mass_kg', 10.0):.1f}kg  type={body_type:<9}  "
          f"{coll_approx}  scale={scale:.4f}  size={sz.get('confidence')}"
          + (f"  [forced: {demotion}]" if demotion else ""), flush=True)

    if not dry_run and not (skip_existing and usd_physics.exists()):
        usd_visual.parent.mkdir(parents=True, exist_ok=True)
        # Open3D decimation occasionally segfaults under memory pressure;
        # a single retry clears it without masking real failures.
        visual_args = [VENV_PYTHON, MESH_TO_USD, glb, "--output", usd_visual,
                       "--scale", str(scale), "--target-faces", str(TARGET_FACES),
                       "--roughness", str(pa.get("roughness", 0.45)),
                       "--metallic", str(pa.get("metallic", 0.0)),
                       "--prim-name", prim_name(oid), "--asset-name", label or oid]
        try:
            _run(visual_args, f"mesh_to_usd {oid}")
        except RuntimeError:
            print(f"  WARNING {oid}: mesh_to_usd failed, retrying once", flush=True)
            _run(visual_args, f"mesh_to_usd {oid} (retry)")

        physics_args = [VENV_PYTHON, CONFIGURE_PHYSICS, usd_visual, "--output", usd_physics,
                        "--mass", str(pa.get("mass_kg", 10.0)),
                        "--collision-approximation", coll_approx, "--body-type", body_type]
        try:
            _run(physics_args, f"configure_usd_physics {oid}")
        except RuntimeError as exc:
            if body_type != "dynamic":
                raise
            # Inertia computation failed (SIGSEGV, assertion, or other).
            # Demote to static so the USD is still written.
            body_type = "static"
            demotion = f"physics configuration failed: {str(exc)[:120]}"
            print(f"  WARNING {oid}: {demotion!r} — retrying as static", flush=True)
            _run(physics_args[:-2] + ["--body-type", "static"],
                 f"configure_usd_physics {oid} (static retry)")

    entry = {"object_id": oid, "status": "success", "generator": "sam3d"}
    entry["generation"] = generation_source(record, generation, handoff)
    if dry_run:
        return entry

    conversion = usd_visual.with_suffix(".conversion.json")
    physics_report = json.loads(usd_physics.with_suffix(".physics.json").read_text())
    body_type = physics_report["body_type"]
    dimensions = json.loads(conversion.read_text())["dimensions_m"]
    entry["usd"] = f"usds/{oid}/physics.usdc"
    entry["size"] = size_block(sz, dimensions)
    pose_path = generation / record["pose"] if record.get("pose") else None
    if pose_path and pose_path.exists() and json.loads(pose_path.read_text()).get("available"):
        entry["pose"] = pose_in_usd_frame(json.loads(pose_path.read_text()), scale,
                                          usd_offset(glb, scale, conversion))
    else:
        entry["pose"] = None
    entry["physics"] = physics_block(pa, body_type, demotion)
    entry["articulation"] = None
    return entry


def build(generation: Path, output: Path, *, dry_run: bool = False,
          object_ids: list[str] | None = None,
          skip_existing: bool = False) -> None:
    generation = generation.resolve()

    manifest_src = json.loads((generation / "manifest.json").read_text())
    handoff = Path(manifest_src["source_handoff"]["path"])
    upstream = {"path": str(handoff),
                "schema_version": manifest_src["source_handoff"].get("schema_version"),
                "manifest_sha256": sha256(handoff / "manifest.json"),
                "objects_sha256": sha256(handoff / "objects.json")}
    # Generation pinned the upstream bytes it read; if they moved since, frames
    # and masks the assets were made from may no longer match the handoff.
    generated_from = {k: manifest_src["source_handoff"].get(k)
                      for k in ("manifest_sha256", "objects_sha256")}
    upstream["matches_generation_input"] = all(
        upstream[k] == v for k, v in generated_from.items())
    if not upstream["matches_generation_input"]:
        upstream["generation_input"] = generated_from

    all_records = json.loads((generation / "objects.json").read_text())
    if object_ids:
        all_records = [r for r in all_records if r["object_id"] in object_ids]

    pa_map = {o["object_id"]: o
              for o in json.loads((generation / "physics_appearance.json").read_text()).get("objects", [])}
    sz_map = {o["object_id"]: o
              for o in json.loads((generation / "sizes.json").read_text()).get("objects", [])}

    output.mkdir(parents=True, exist_ok=True)
    out_objects = []
    for i, record in enumerate(sorted(all_records, key=lambda r: r["object_id"]), 1):
        oid = record["object_id"]
        if record["status"] != "success":
            # Kept so assembly can tell "not generated" from "dropped by mistake".
            out_objects.append({"object_id": oid, "status": record["status"],
                                "reason": record.get("reason") or record.get("error"),
                                "generation": {"view": record.get("view"),
                                               **(record.get("source") or {})}})
            continue
        print(f"[{i:2d}/{len(all_records)}] ", end="")
        if record.get("generator") == "dualcraft":
            out_objects.append(build_dualcraft_object(record, generation, output,
                                                      pa_map.get(oid, {}), sz_map.get(oid, {}),
                                                      dry_run=dry_run, skip_existing=skip_existing))
            continue
        out_objects.append(build_object(record, generation, handoff, output,
                                        pa_map.get(oid, {}), sz_map.get(oid, {}),
                                        dry_run=dry_run, skip_existing=skip_existing))
    if dry_run:
        return

    # objects.json — merge with prior results so --object-id re-runs don't
    # drop the objects built by earlier batches.
    objects_path = output / "objects.json"
    merged: dict[str, dict] = {}
    if objects_path.exists() and object_ids:
        merged = {o["object_id"]: o for o in json.loads(objects_path.read_text())}
    for entry in out_objects:
        merged[entry["object_id"]] = entry
    ordered = sorted(merged.values(), key=lambda o: o["object_id"])
    objects_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")

    status_counts: dict[str, int] = {}
    for o in ordered:
        status_counts[o["status"]] = status_counts.get(o["status"], 0) + 1
    built = [o for o in ordered if o["status"] == "success"]
    upstream["generation_inputs_unchanged"] = (
        f"{sum(1 for o in built if o['generation'].get('inputs_match_handoff'))}/{len(built)} "
        "assets' source RGB and mask still match handoff/v1 byte for byte")
    manifest_out = {
        "schema_version": "assembly_handoff/v2",
        "scene_id": manifest_src.get("scene_id", ""),
        "source_handoff": upstream,
        "source_generation": {
            "path": str(generation),
            "schema_version": manifest_src.get("schema_version"),
            "generator": manifest_src.get("generator"),
            "manifest_sha256": sha256(generation / "manifest.json"),
        },
        "join": "objects.json[].object_id == handoff/v1 objects.json[].object_id; "
                "labels, clouds, key frames, camera, centroid and bounds live in handoff/v1",
        "asset_frame": "USD Z-up, metres, defaultPrim at the origin; mesh XY bounding box "
                       "centred on the origin and lowest point at z=0",
        "conventions": {"pose": POSE_CONVENTION, "pose_scene": SCENE_POSE_CONVENTION,
                        "articulation": ARTICULATION_CONVENTION, "size": SIZE_CONVENTION},
        "generator_counts": {g: sum(1 for o in built if o.get("generator", "sam3d") == g)
                             for g in sorted({o.get("generator", "sam3d") for o in built})},
        "status_counts": status_counts,
        "objects": len(ordered),
        "updated_at": datetime.datetime.now().astimezone().isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(out_objects)} records this run, {len(ordered)} total "
          f"{status_counts} -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("generation", type=Path,
                        help="Generation output directory (contains manifest.json)")
    parser.add_argument("--output", type=Path,
                        help="Assembly handoff output directory "
                             "(default: <generation>/assembly_handoff)")
    parser.add_argument("--object-id", action="append", dest="object_ids",
                        help="Repeat to process a subset; default is every generation record")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running USD conversion")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse USDs already on disk and only refresh objects.json")
    args = parser.parse_args()

    generation = args.generation.resolve()
    output = args.output.resolve() if args.output else generation / "assembly_handoff"
    build(generation, output, dry_run=args.dry_run, object_ids=args.object_ids,
          skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
