"""Build the step-by-step result page for structure generation.

Reads one structuregen run (``output/<scene>/structuregen/<stamp>/``):

    plan.json          segments, prisms and expected volumes (stage 1 output)
    build_report.json  what Blender built: closedness, volume, zones, cameras
    structure_base.usda  the white model: /Structure/Walls and /Structure/Floors
    views/*.png        renders from every key-frame camera, plus an overview

    evidence/          what the agent is given (stage 2)
    agent_<stamp>/     one appearance-agent run (stage 3): report.json, usage.jsonl,
                       session/transcript.jsonl, session/compare/, final_compare/, appearance.blend

The newest non-fake agent run is shown unless ``--agent`` names one. Its model is read back
from appearance.blend by Blender (src/structuregen/export/blender_dump.py), so the colours and
trims on the page are what the file holds, not what the agent said it did. Meshes are
embedded as flat triangles and images copied beside the page, so it opens over ``file://``
with no server and no network.

Usage::

    PYTHONPATH=src .venv-sam3d/bin/python pagegen/build_structuregen_page.py [run dir]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "pagegen" / "structuregen"
DEFAULT_OUTPUT = ROOT / "site" / "structuregen"
LATEST = ROOT / "output/scene0000_00/structuregen/latest"

sys.path.insert(0, str(ROOT / "src"))
from structuregen.config import BLENDER, SEGMENT_COLOURS as PALETTE  # noqa: E402  same colours as the plan view
from structuregen.common import elevation as el  # noqa: E402

DUMP = ROOT / "src" / "structuregen" / "export" / "blender_dump.py"
FACES = ("front", "reveal", "back", "top", "bottom", "end_start", "end_end")
THUMB_WIDTH = 640


def triangles(stage_path: Path, colours: dict[str, tuple[int, int, int]]):
    """Flat triangle soup, one colour per segment (the same colours as the table)."""
    stage = Usd.Stage.Open(str(stage_path))
    positions, normals, colours_out = [], [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        segment = prim.GetParent().GetName()
        floor = prim.GetParent().GetParent().GetName() == "Floors"
        colour = np.array((218, 212, 200) if floor else colours.get(segment, (170, 170, 170)), np.float32) / 255.0
        # USD matrices act on row vectors: p_world = [p, 1] @ M
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        pts = np.array(mesh.GetPointsAttr().Get(), np.float64)
        pts = (np.c_[pts, np.ones(len(pts))] @ xf)[:, :3]
        counts = list(mesh.GetFaceVertexCountsAttr().Get())
        idx = list(mesh.GetFaceVertexIndicesAttr().Get())
        k = 0
        for c in counts:
            face = idx[k:k + c]
            k += c
            for t in range(1, c - 1):
                tri = pts[[face[0], face[t], face[t + 1]]]
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                length = np.linalg.norm(n)
                n = n / length if length else np.array([0.0, 0.0, 1.0])
                for vertex in tri:
                    positions.append(vertex)
                    normals.append(n)
                    colours_out.append(colour)
    positions = np.array(positions, np.float32)
    return positions, np.array(normals, np.float32), np.array(colours_out, np.float32)


def outline(render: Path, target: Path) -> None:
    """The model's edges as a transparent PNG: silhouette plus creases between faces.

    Workbench adds cavity speckle on flat faces, so edges come from a blurred image and a
    high Canny threshold; the silhouette comes from alpha and is always kept.
    """
    rgba = cv2.imread(str(render), cv2.IMREAD_UNCHANGED)
    alpha = rgba[..., 3]
    grey = cv2.GaussianBlur(cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY), (0, 0), 2.0)
    creases = cv2.Canny(grey, 18, 40) & (cv2.erode(alpha, np.ones((5, 5), np.uint8)) > 0)
    silhouette = cv2.Canny(alpha, 50, 150)
    edges = cv2.dilate(((creases | silhouette) > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    out = np.zeros((*alpha.shape, 4), np.uint8)
    out[edges > 0] = (0, 225, 255, 255)        # BGRA: amber
    cv2.imwrite(str(target), out)


def evidence_payload(run: Path, plan: dict, handoff: Path, out: Path) -> dict | None:
    """Stage 2 as the agent gets it: task card, plan view, the first wall's frames. None until it exists."""
    ev = run / "evidence"
    if not (ev / "task_card.json").is_file():
        return None
    (out / "input").mkdir(exist_ok=True)
    shutil.copyfile(ev / "plan.png", out / "input/plan.png")
    card = json.loads((ev / "task_card.json").read_text())
    first = []
    for rel in card["first"]["files"]:
        name = Path(rel).name
        shutil.copyfile(ev / rel, out / "input" / name)
        first.append(name)
    pairs = [{"frame": int(f), "rgb": f"input/{f}_rgb.jpg", "boxed": f"input/{f}_boxed.jpg"}
             for f in dict.fromkeys(n.split("_")[0] for n in first)]
    text = (ev / "task_card.md").read_text()
    return {"card": card, "card_text": text, "card_chars": len(text), "plan": "input/plan.png",
            "first_pairs": pairs, "ranking": json.loads((ev / "first_wall.json").read_text())}


def latest_agent(run: Path) -> Path | None:
    """The newest real (not --fake) agent run that wrote a report."""
    runs = sorted(d for d in run.glob("agent_2*") if (d / "report.json").is_file() and (d / "usage.jsonl").is_file())
    return runs[-1] if runs else None


def thumb(src: Path, dst: Path) -> None:
    img = cv2.imread(str(src))
    if img.shape[1] > THUMB_WIDTH:
        img = cv2.resize(img, (THUMB_WIDTH, round(img.shape[0] * THUMB_WIDTH / img.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def read_model(blend: Path) -> dict:
    """appearance.blend as Blender holds it: triangles, face colours, trim properties."""
    target = blend.parent / "page_dump.json"
    result = subprocess.run([str(BLENDER), "-b", str(blend), "--python", str(DUMP), "--", str(target)],
                            capture_output=True, text=True)
    if result.returncode or not target.is_file():
        raise SystemExit(f"Blender dump failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    doc = json.loads(target.read_text())
    target.unlink()
    return doc


def model_mesh(doc: dict) -> dict:
    """Flat triangles coloured with the sRGB value the agent gave each face."""
    positions, normals, colours = [], [], []
    for ob in doc["objects"]:
        for t in ob["triangles"]:
            slot = ob["slots"][t["m"]] if t["m"] < len(ob["slots"]) else {}
            rgb = np.array(slot.get("rgb") or (200, 200, 200), np.float32) / 255.0
            tri = np.array(t["p"], np.float64)
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            n = n / (np.linalg.norm(n) or 1.0)
            for v in tri:
                positions.append(v)
                normals.append(n)
                colours.append(rgb)
    positions = np.array(positions, np.float32)
    return {"position": b64(positions), "normal": b64(np.array(normals)), "colour": b64(np.array(colours)),
            "count": int(len(positions))}


def photo_colour(handoff: Path, walls_doc: dict, seg: dict, frames: list[int]) -> list[int] | None:
    """Median photo colour of the segment's room-side face where the handoff mask says the wall is seen.

    The face is sampled every 4 cm, projected into each of the wall's key frames, and the pixels
    inside the mask are pooled. Lit colour, not paint colour: a reference only.
    """
    wall = next(w for w in walls_doc["walls"] if w["id"] == seg["wall"])
    keys = {k["source_frame_id"]: k for k in wall["key_frames"]}
    cam = walls_doc["camera"]
    a = np.arange(seg["a0"] + 0.02, seg["a1"], 0.04)
    z = np.arange(0.3, seg["height_m"] - 0.1, 0.04)          # off the skirting and the ceiling line
    A, Z = np.meshgrid(a, z)
    pts = el.wall_point(wall, A, Z)
    room_side = -np.array(wall["normal_to_back"][:2], float)
    pooled = []
    for fid in frames:
        k = keys.get(fid)
        if not k:
            continue
        c2w = np.array(k["camera_to_world"], float)
        img = cv2.imread(str(handoff / k["rgb"]))
        mask = cv2.imread(str(handoff / k["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        u, v, d = el.project(cam, c2w, pts)
        facing = ((c2w[:2, 3] - pts[..., :2]) @ room_side) > 0
        ok = (d > 0.05) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"]) & facing
        ui, vi = u[ok].astype(int), v[ok].astype(int)
        keep = mask[vi, ui]
        pooled.append(img[vi[keep], ui[keep]][:, ::-1])
    pooled = np.concatenate(pooled) if pooled else np.zeros((0, 3))
    return [int(c) for c in np.median(pooled, 0)] if len(pooled) >= 20 else None


def agent_payload(agent: Path | None, plan: dict, handoff: Path, walls_doc: dict, out: Path) -> dict | None:
    """Stage 3 result: one agent run, read from its own files and from appearance.blend."""
    if agent is None:
        return None
    report = json.loads((agent / "report.json").read_text())
    usage = [json.loads(line) for line in (agent / "usage.jsonl").read_text().splitlines() if line.strip()]
    steps = [json.loads(line) for line in (agent / "session/transcript.jsonl").read_text().splitlines() if line.strip()]
    sources = [{"dir": str(agent), "replies": report["replies"], "usd": report["usd"], "what": "this run"}]
    if report.get("resumed_from"):
        # walls replayed from an earlier run: show that run's wall steps and cost alongside this one's
        replay = Path(report["resumed_from"])
        src = Path(json.loads((replay / "report.json").read_text())["replayed_from"])
        src_report = json.loads((src / "report.json").read_text())
        src_usage = [json.loads(l) for l in (src / "usage.jsonl").read_text().splitlines() if l.strip()]
        src_steps = [json.loads(l) for l in (src / "session/transcript.jsonl").read_text().splitlines() if l.strip()]
        kept = {w["wall"] for w in report["walls"] if w.get("replayed")}
        src_usage = [r for r in src_usage if r["wall"] in kept]
        src_steps = [x for x in src_steps if x["segment"] in kept]
        for r in src_usage:
            r["n"] = f"w{r['n']}"
        for x in src_steps:
            x["n"] = f"w{x['n']}"
        reps = {w["wall"]: w["replies"] for w in src_report["walls"]}
        for w in report["walls"]:
            if w.get("replayed"):
                w["replies"] = reps.get(w["wall"], 0)
        for sid in kept:
            report["wall_state"].setdefault(sid, src_report["wall_state"][sid])
        sources.insert(0, {"dir": str(src), "replies": sum(reps.get(k, 0) for k in kept),
                           "usd": round(sum(r["cost"] for r in src_usage), 4), "what": "walls, replayed"})
        usage, steps = src_usage + usage, src_steps + steps
    order = [w["wall"] for w in report["walls"]]
    seg_of = {s["id"]: s for s in plan["segments"]}
    number = {s["id"]: i + 1 for i, s in enumerate(plan["segments"])}
    floor_of = {f["id"]: f for f in plan.get("floors", [])}
    media = out / "agent"
    for sub in ("steps", "final", "photos"):
        (media / sub).mkdir(parents=True, exist_ok=True)

    # Which reply made each tool call: replies list their calls in order, the transcript logs
    # them in the same order (opening briefs are the harness's own, not a reply's).
    queue = [(r["n"], name) for r in usage for name in r["calls"]]
    for st in steps:
        if st["tool"] == "finish_segment" and isinstance(st["args"].get("n"), int):
            st["segment"] = next(sid for sid, k in number.items() if k == st["args"]["n"])   # logged under the next wall before the fix
        elif st["tool"] == "finish_segment" and st["args"].get("n") in floor_of:
            st["segment"] = st["args"]["n"]
        if st["tool"] == "commit":
            st["segment"] = "wrapup"
        if st["tool"] == "brief":
            st["reply"] = None
            continue
        for i, (n, name) in enumerate(queue):
            if name == st["tool"]:
                st["reply"] = n
                del queue[:i + 1]
                break

    def copy_step_image(path: str | None) -> str | None:
        if not path or not Path(path).is_file():
            return None
        shutil.copyfile(path, media / "steps" / Path(path).name)
        return f"agent/steps/{Path(path).name}"

    def copy_photo(path: str) -> str:
        p = Path(path)
        name = f"{p.parent.name}_{p.name}"
        if not (media / "photos" / name).is_file():
            thumb(p, media / "photos" / name)
        return f"agent/photos/{name}"

    timeline = {w: [] for w in order}
    for st in steps:
        r = st["result"]
        entry = {"n": st["n"], "tool": st["tool"], "reply": st.get("reply"), "args": st["args"],
                 "text": r.get("text", ""), "refused": bool(r.get("refused")),
                 "rolled_back": bool(r.get("rolled_back")), "trims": r.get("trims", []),
                 "image": copy_step_image(r.get("image")),
                 "photos": [copy_photo(p) for p in r.get("images", [])]}
        timeline.setdefault(st["segment"], []).append(entry)

    doc = read_model(agent / "appearance.blend")
    objects = {o["name"]: o for o in doc["objects"]}
    walls = []
    for w in report["walls"]:
        sid = w["wall"]
        rows = [r for r in usage if r["wall"] == sid]
        entry = {"segment": sid, "status": w["status"], "replies": w["replies"],
                 "usd": round(sum(r["cost"] for r in rows), 4),
                 "seconds": round(sum(r["seconds"] for r in rows), 1), "steps": timeline.get(sid, [])}
        if sid in seg_of:
            seg, state = seg_of[sid], report["wall_state"][sid]
            ob = objects[sid]
            faces = {}
            for slot in ob["slots"]:
                face = slot["name"][len(sid) + 1:]
                faces[face] = {"rgb": slot["rgb"], "item": slot["item"], "roughness": slot["roughness"]}
            trims = []
            for child in (o for o in doc["objects"] if o["parent"] == sid):
                _, kind, item = child["name"].split("__")
                p = child["props"]
                trims.append({"name": child["name"], "kind": kind, "item": item, "rgb": child["slots"][0]["rgb"],
                              "z": p.get("sg_z"), "along": p.get("sg_along"), "depth": p.get("sg_depth"),
                              "width": p.get("sg_width"), "sides": p.get("sg_sides")})
            entry.update(number=number[sid], role=seg["role"], length_m=seg["length_m"],
                         openings=len(seg["openings"]), frames=state["frames"], edits=state["edits"],
                         declared=state["declared"], items=state["items"], summary=state.get("summary"),
                         faces=faces, trims=trims,
                         photo_rgb=photo_colour(handoff, walls_doc, seg, state["frames"]))
        elif sid in floor_of:
            f, state = floor_of[sid], report["wall_state"][sid]
            ob = objects[sid]
            faces = {slot["name"][len(sid) + 1:]: {"rgb": slot["rgb"], "item": slot["item"], "roughness": slot["roughness"],
                                                   "tiles": slot.get("tiles")} for slot in ob["slots"]}
            entry.update(floor=True, kind=f["kind"], area_m2=f["area_m2"], frames=state["frames"], edits=state["edits"],
                         declared=state["declared"], items=state["items"], summary=state.get("summary"), faces=faces)
        walls.append(entry)

    final = []
    for f in sorted((agent / "final_compare").glob("*.jpg")):
        shutil.copyfile(f, media / "final" / f.name)
        fid = int(f.stem)
        final.append({"frame": fid, "image": f"agent/final/{f.name}",
                      "walls": [s["id"] for s in plan["segments"] if fid in report["wall_state"].get(s["id"], {}).get("frames", [])]})

    tokens = {k: sum(r["usage"].get(k, 0) for r in usage) for k in ("input_tokens", "output_tokens")}
    tokens["cached_tokens"] = sum(r["usage"].get("input_tokens_details", {}).get("cached_tokens", 0) for r in usage)
    tokens["max_input"] = max((r["usage"]["input_tokens"] for r in usage), default=0)
    tokens["reasoning_tokens"] = sum(r["usage"].get("output_tokens_details", {}).get("reasoning_tokens", 0) for r in usage)
    tools = {}
    for st in steps:
        if st["tool"] != "brief":
            tools[st["tool"]] = tools.get(st["tool"], 0) + 1
    return {"dir": str(agent), "sources": sources, "status": report["status"], "model": report["model"],
            "served_by": next((r.get("model") for r in usage if r.get("model")), None),
            "replies": len(usage), "usd": round(sum(r["cost"] for r in usage), 4),
            "seconds": round(sum(r["seconds"] for r in usage)),
            "tool_calls": sum(tools.values()), "tools": tools, "tokens": tokens,
            "usage": [{"n": r["n"], "wall": r["wall"], "cost": r["cost"], "seconds": r["seconds"],
                       "input": r["usage"]["input_tokens"], "output": r["usage"]["output_tokens"],
                       "cached": r["usage"].get("input_tokens_details", {}).get("cached_tokens", 0),
                       "calls": r["calls"]} for r in usage],
            "walls": walls, "final": final, "faces": FACES,
            "mesh": model_mesh(doc)}


def b64(array: np.ndarray) -> str:
    return base64.b64encode(array.astype(np.float32).tobytes()).decode("ascii")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?", default=LATEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--agent", type=Path, help="Agent run directory; default the newest real run in <run>")
    args = parser.parse_args(argv)
    run = Path(args.run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    report = json.loads((run / "build_report.json").read_text())
    colour_of = {seg["id"]: PALETTE[i % len(PALETTE)] for i, seg in enumerate(plan["segments"])}
    usd = run / "structure_base.usda" if (run / "structure_base.usda").is_file() else run / "walls_base.usda"
    positions, normals, colours = triangles(usd, colour_of)

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    (out / "views").mkdir(parents=True)

    # Key frames: the handoff photo, the model rendered from the same camera, and which
    # walls the handoff gave this frame for.
    handoff = Path(plan["handoff_dir"])
    walls_doc = json.loads((handoff / "walls.json").read_text())
    frames = {}
    for w in walls_doc["walls"]:
        for k in w["key_frames"]:
            f = frames.setdefault(k["source_frame_id"], {"fid": k["source_frame_id"], "photo": k["rgb"],
                                                         "walls": [], "angle": k.get("view_angle_deg")})
            f["walls"].append(w["id"])
    views = []
    for fid, f in sorted(frames.items()):
        render = run / "views" / f"frame_{fid:04d}.png"
        if not render.is_file():
            continue
        shutil.copyfile(handoff / f["photo"], out / "views" / f"photo_{fid:04d}.jpg")
        shutil.copyfile(render, out / "views" / f"model_{fid:04d}.png")
        outline(render, out / "views" / f"edges_{fid:04d}.png")
        views.append({"fid": fid, "photo": f"views/photo_{fid:04d}.jpg", "model": f"views/model_{fid:04d}.png",
                      "edges": f"views/edges_{fid:04d}.png",
                      "walls": f["walls"]})
    built = {s["id"]: s for s in report["segments"]}
    segments = []
    for seg in plan["segments"]:
        b = built.get(seg["id"], {})
        segments.append({
            "id": seg["id"], "wall": seg["wall"], "role": seg["role"],
            "a0": seg["a0"], "a1": seg["a1"], "length_m": seg["length_m"],
            "thickness_m": seg["thickness_m"], "height_m": seg["height_m"],
            "expected_volume_m3": seg["expected_volume_m3"],
            "volume_m3": b.get("volume_m3"), "closed": b.get("closed"),
            "faces": b.get("faces"), "zones": b.get("zones", []),
            "openings": len(seg["openings"]),
            "colour": "#%02x%02x%02x" % colour_of[seg["id"]],
        })

    payload = {
        "schema": "structuregen/page/v1",
        "scene": plan["scene_id"],
        "run": str(run),
        "height_m": plan["height_m"],
        "cuts": plan["cuts"],
        "dropped_openings": plan.get("dropped_openings", []),
        "blender": report.get("blender"),
        "frames": report.get("frames", []),
        "segments": segments,
        "views": views,
        "evidence": evidence_payload(run, plan, handoff, out),
        "agent": agent_payload(args.agent.resolve() if args.agent else latest_agent(run), plan, handoff, walls_doc, out),
        "mesh": {"position": b64(positions), "normal": b64(normals), "colour": b64(colours),
                 "count": int(len(positions))},
    }
    for name in ("index.html", "page.css", "page.js"):
        shutil.copyfile(TEMPLATES / name, out / name)
    html = (out / "index.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/", "window.STRUCTUREGEN = " + json.dumps(payload, ensure_ascii=False) + ";")
    (out / "index.html").write_text(html, encoding="utf-8")
    total = sum(s["volume_m3"] or 0 for s in segments)
    print(f"wrote {out / 'index.html'} ({len(segments)} segments, {len(views)} views, "
          f"{len(positions)} triangles, wall volume {total:.3f} m3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
