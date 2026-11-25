"""Build a self-contained review page for one DualCraft run.

Reads a local copy of the run (``output/dualcraft/<run>/``: inputs, product URDF, the
record's renders, ledger and trajectory, plus ``viewer_links/`` from
``dualcraft_bake_links.py``) and writes one HTML file with every image and mesh embedded,
so it opens over file://.

The 3D view poses the URDF: each link's mesh is placed by the joint chain, and the
revolute joints can be swung with sliders.

Usage::

    .venv-sam3d/bin/python pagegen/build_dualcraft_run_page.py \\
        output/dualcraft/scannet_s0000_obj0038_v2 --issues docs/dualcraft/obj0038_issues.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "pagegen" / "dualcraft_run" / "index.html"
HANDOFF_KEYFRAMES = ROOT.parent / "InstanceRecognition" / "artifacts" / "handoff" / "scene0000_00" / "gt"


def image_uri(path: Path, max_side: int = 900, quality: int = 82) -> str:
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "P"):
        # Blender renders have a transparent background; JPEG would turn it black.
        image = image.convert("RGBA")
        backdrop = Image.new("RGBA", image.size, (233, 236, 239, 255))
        image = Image.alpha_composite(backdrop, image)
    image = image.convert("RGB")
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def floats(text: str | None, default=(0.0, 0.0, 0.0)) -> list[float]:
    return [float(v) for v in text.split()] if text else list(default)


def read_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        links[link.get("name")] = {
            "mass": float(inertial.find("mass").get("value")) if inertial is not None else None,
        }
    joints = []
    for joint in root.findall("joint"):
        origin, axis, limit = joint.find("origin"), joint.find("axis"), joint.find("limit")
        joints.append({
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "xyz": floats(origin.get("xyz") if origin is not None else None),
            "rpy": floats(origin.get("rpy") if origin is not None else None),
            "axis": floats(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            "lower": float(limit.get("lower")) if limit is not None else 0.0,
            "upper": float(limit.get("upper")) if limit is not None else 0.0,
        })
    return {"name": root.get("name"), "links": links, "joints": joints}


PIECE_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 200, 200],
    [240, 50, 230], [160, 190, 30], [250, 170, 190], [0, 128, 128], [170, 110, 40], [128, 0, 0],
    [120, 200, 150], [128, 128, 0], [255, 200, 120], [0, 0, 128], [128, 128, 128], [200, 90, 90],
    [90, 90, 200], [220, 200, 30],
], dtype=np.uint8)


def encode_link_mesh(path: Path) -> dict:
    mesh = trimesh.load(path, process=False)
    colours = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
    # Flat-shaded panels read better than smoothed normals on boxy furniture.
    vertices = np.asarray(mesh.vertices, dtype=np.float32)[mesh.faces].reshape(-1, 3)
    normals = np.repeat(np.asarray(mesh.face_normals, dtype=np.float32), 3, axis=0)
    face_colours = colours[mesh.faces].reshape(-1, 3)
    # Connected pieces (vertices welded by position), the same count the compile QC reports
    # as "disconnected geometry islands".
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    labels = trimesh.graph.connected_component_labels(welded.face_adjacency, node_count=len(welded.faces))
    piece_colours = np.repeat(PIECE_PALETTE[labels % len(PIECE_PALETTE)], 3, axis=0)
    return {
        "positions": b64(vertices),
        "normals": b64(normals),
        "colours": b64(face_colours),
        "piece_colours": b64(piece_colours),
        "count": int(len(vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(labels.max() + 1),
    }


def harness_events(trajectory: Path) -> list[dict]:
    """The harness's own bookkeeping events, one line of Chinese each, keyed by turn."""
    events = []
    for line in trajectory.read_text().splitlines():
        r = json.loads(line)
        kind, turn = r["type"], r.get("turn")
        if kind == "diff_ledger_auto_open":
            text = f"自动开出待办 {', '.join(r['ids'])}：四个门的关节轴只是占位（默认 +z），要求 agent 用 pose_check 确认"
            turn = 0
        elif kind == "diff_ledger_review":
            bits = []
            if r["opened"]: bits.append("新开 " + ", ".join(r["opened"]))
            if r["resolved"]: bits.append("关闭 " + ", ".join(r["resolved"]))
            if r["dropped"]: bits.append("放弃 " + ", ".join(r["dropped"]))
            text = f"差异复查（第 {r['revision']} 版）：" + ("；".join(bits) or "无变化") + f"；剩 {r['open_blocking']} 条阻塞"
        elif kind == "pivots_placed":
            text = "门的铰链枢轴放到外侧竖梃：" + ", ".join(r["placed"])
        elif kind == "joint_axis_set":
            changed = [f"{name.removesuffix('_joint')} {j['was']}→{'+' if j['axis'][2] > 0 else '-'}z"
                       for name, j in r["joints"].items() if j["changed"]]
            text = "改关节轴：" + ", ".join(changed) + "（其余两扇确认 +z）"
        elif kind == "diff_loop_phase_switch":
            text = f"几何阶段结束（{r['reason']}，共 {r['edit_count']} 次编辑），进入 B 阶段：验证 → 提交 → 清单 → 编译"
        elif kind == "diff_loop_preflight":
            text = "提交预检未过：" + " / ".join(r["blocking"]) + "，打开 finish_gate 返工窗口（3 轮）"
        elif kind == "diff_loop_rework_expired":
            text = f"返工窗口过期（第 {r['opened_turn']} 轮打开），几何未改"
        elif kind == "harness_finish_attempt":
            text = "compile_model(finalize) " + ("被接受，收工" if r["accepted"] else "被拒绝")
        else:
            continue
        events.append({"turn": turn, "type": kind, "text": text})
    return events


def find(directory: Path, pattern: str) -> Path:
    hits = sorted(directory.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"{directory}/{pattern}")
    return hits[0]


def build(run: Path, issues_path: Path, output: Path) -> None:
    renders = run / "rev" / "blender" / "renders"
    job = json.loads((run / "job.json").read_text())
    handoff = json.loads((run / "inputs" / "scannet_handoff.json").read_text())
    blueprint = json.loads((run / "inputs" / "prompt.json").read_text())
    source = json.loads((run / "inputs" / "blueprint" / "source.json").read_text())
    cost = json.loads((run / "rev" / "cost.json").read_text())
    report = json.loads((run / "product" / "compile_report.json").read_text())
    urdf = read_urdf(run / "product" / "model.urdf")
    turns = json.loads((run / "turns.json").read_text())
    issues = json.loads(issues_path.read_text())

    ledger = []
    for line in (run / "rev" / "blender" / "diff_ledger.jsonl").read_text().splitlines():
        event = json.loads(line)
        item = event.get("item")
        if event["event"] == "open" and item:
            ledger.append({"id": item["id"], "kind": item["kind"], "view": item["view"],
                           "parts": item["parts"], "note": item["note"], "opened_turn": event.get("turn"),
                           "status": "open"})
        elif event["event"] in ("resolve", "drop") and item:
            for entry in ledger:
                if entry["id"] == item["id"]:
                    entry["status"] = "resolved" if event["event"] == "resolve" else "dropped"
                    entry["closed_turn"] = event.get("turn")

    # Key frames with their mask box, as the inputs were cut from them.
    keyframes = []
    used = {v["keyframe"] for v in handoff["views"]}
    for view in handoff["views"] + handoff.get("unused_views", []):
        seq = f"{view['keyframe']:04d}"
        boxed = HANDOFF_KEYFRAMES / "keyframes" / handoff["object"] / f"{seq}_boxed.jpg"
        crop = run / "inputs" / "reference" / view["file"]
        keyframes.append({
            "seq": view["keyframe"],
            "used": view["keyframe"] in used,
            "file": view["file"],
            "azimuth": view.get("azimuth_deg"),
            "elevation": view.get("elevation_deg"),
            "reason": view.get("reason"),
            "boxed": image_uri(boxed, 700) if boxed.exists() else None,
            "crop": image_uri(crop, 600) if crop.exists() else None,
        })

    images = {
        "compare0": image_uri(find(renders, "compare-revision-0004-ref_photo_0-*.png"), 1300),
        "compare1": image_uri(find(renders, "compare-revision-0004-ref_photo_1-*.png"), 1300),
        "front": image_uri(find(renders, "revision-0004-front-*.png"), 600),
        "front_left": image_uri(find(renders, "revision-0004-front_left-13f*.png"), 600),
        "front_right": image_uri(find(renders, "revision-0004-front_right-*.png"), 600),
        "top": image_uri(find(renders, "revision-0004-top-*.png"), 600),
        "rev1_front": image_uri(find(renders, "revision-0001-front-*.png"), 600),
        "rev1_left": image_uri(find(renders, "revision-0001-left_straight-*.png"), 600),
        "pose_t07_door_2": image_uri(renders / "pose-check-turn07-door_2_joint.png", 1100),
        "pose_t10_door_2": image_uri(renders / "pose-check-turn10-door_2_joint.png", 1100),
        "pose_t07_door_1": image_uri(renders / "pose-check-turn07-door_1_joint.png", 1100),
        "pose_t07_door_3": image_uri(renders / "pose-check-turn07-door_3_joint.png", 1100),
        "pose_t07_door_4": image_uri(renders / "pose-check-turn07-door_4_joint.png", 1100),
        "pose_t10_door_4": image_uri(renders / "pose-check-turn10-door_4_joint.png", 1100),
    }

    links_dir = run / "viewer_links"
    link_info = json.loads((links_dir / "info.json").read_text())
    meshes = {name: encode_link_mesh(links_dir / f"{name}.ply") for name in urdf["links"]}
    for name, info in link_info.items():
        urdf["links"][name].update(pieces=info["pieces"], bounds=info["bounds"])

    history = {h["stage"]: h["at"] for h in job["history"]}
    payload = {
        "run": run.name,
        "job": {"settings": job["settings"], "history": history, "relay_mode": job.get("relay_mode")},
        "handoff": handoff,
        "keyframes": keyframes,
        "blueprint": blueprint,
        "blueprint_usage": source.get("usage"),
        "estimates": source.get("estimates"),
        "cost": cost["total"],
        "compile": {"status": report["status"], "warnings": [w["message"] for w in report.get("warnings", [])],
                    "turns": report["metrics"]["turn_count"],
                    "tool_calls": report["metrics"]["tool_call_count"],
                    "compiles": report["metrics"]["compile_attempt_count"]},
        "urdf": urdf,
        "meshes": meshes,
        "turns": turns,
        "ledger": ledger,
        "events": harness_events(run / "rev" / "traces" / "trajectory.jsonl"),
        "images": images,
        "issues": issues,
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="local copy of the run, output/dualcraft/<run>")
    parser.add_argument("--issues", type=Path, required=True, help="issues JSON for this run")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: docs/dualcraft/<run>.html")
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output or ROOT / "docs" / "dualcraft" / f"{run.name}.html"
    build(run, args.issues.resolve(), output)


if __name__ == "__main__":
    main()
