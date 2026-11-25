"""Build the page describing how objects are generated and exported.

The page explains both generation routes (SAM3D in production, DualCraft on trial) and
the export chain that turns a generated mesh into the assembly handoff.  Every count on
it is read from the generation output and the DualCraft run copies, so rebuilding after
a new run refreshes the numbers; the worked example (one object through both routes)
embeds its images, so the page opens over file://.

Usage::

    .venv-sam3d/bin/python pagegen/build_generation_page.py output/scene0000_00/gt
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "pagegen" / "generation" / "index.html"
DEFAULT_OUTPUT = ROOT / "site" / "generation" / "index.html"
DUALCRAFT_RUNS = ROOT / "output" / "dualcraft"


def image_uri(path: Path, max_side: int = 900) -> str | None:
    if not path.exists():
        return None
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        backdrop = Image.new("RGBA", image.size, (233, 236, 239, 255))
        image = Image.alpha_composite(backdrop, image)
    image = image.convert("RGB")
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def generation_stats(generation: Path) -> dict:
    manifest = read_json(generation / "manifest.json", {})
    handoff = Path(manifest.get("source_handoff", {}).get("path", ""))
    scene = read_json(handoff / "objects.json", [])
    records = read_json(generation / "objects.json", [])
    reports = [read_json(generation / r["report"]) for r in records if r.get("report")]
    reports = [r for r in reports if r]
    seconds = [r["seconds"] for r in reports if r.get("seconds")]
    vram = [r["peak_vram_gib"] for r in reports if r.get("peak_vram_gib")]
    sizes = read_json(generation / "sizes.json", {}).get("objects", [])
    physics = read_json(generation / "physics_appearance.json", {}).get("objects", [])

    assembly_dir = generation / "assembly_handoff"
    assembly = [o for o in read_json(assembly_dir / "objects.json", []) if o.get("status") == "success"]
    reports_phys = [read_json(assembly_dir / "usds" / o["object_id"] / "physics.physics.json") for o in assembly]
    reports_phys = [r for r in reports_phys if r]
    collision_faces = [r["collision_faces"] for r in reports_phys if r.get("collision_faces")]

    def source_kind(source: str) -> str:
        return "footprint" if source.startswith("scan footprint") else source

    return {
        "path": str(generation.relative_to(ROOT)) if generation.is_relative_to(ROOT) else str(generation),
        "scene_id": manifest.get("scene_id"),
        "handoff": {
            "path": str(handoff),
            "schema": manifest.get("source_handoff", {}).get("schema_version"),
            "objects": len(scene),
            "structural": sum(1 for o in scene if o.get("structural")),
            "key_frames": sum(len(o.get("key_frames", [])) for o in scene),
        },
        "sam3d": {
            "status": dict(Counter(r["status"] for r in records)),
            "seconds_median": round(statistics.median(seconds), 1) if seconds else None,
            "seconds_max": round(max(seconds), 1) if seconds else None,
            "minutes_total": round(sum(seconds) / 60, 1) if seconds else None,
            "vram_median": round(statistics.median(vram), 1) if vram else None,
            "vram_max": round(max(vram), 1) if vram else None,
            "watertight": sum(1 for r in reports if r.get("watertight")),
            "faces_median": int(statistics.median(r["faces"] for r in reports)) if reports else None,
            "generated": len(reports),
            "frames": manifest.get("frames", {}),
        },
        "sizes": {
            "total": len(sizes),
            "source": dict(Counter(source_kind(str(s.get("source", ""))) for s in sizes)),
            "confidence": dict(Counter(s.get("confidence") for s in sizes)),
        },
        "physics": {
            "total": len(physics),
            "model": physics and read_json(generation / "physics_appearance.json", {}).get("model"),
            "confidence": dict(Counter(p.get("confidence") for p in physics)),
            "body_type": dict(Counter(p.get("body_type") for p in physics)),
        },
        "assembly": {
            "schema": read_json(assembly_dir / "manifest.json", {}).get("schema_version"),
            "exported": len(assembly),
            "body_type": dict(Counter(o["physics"]["body_type"] for o in assembly)),
            "demoted": sum(1 for o in assembly if o["physics"].get("demotion_reason")),
            "collision": dict(Counter(o["physics"]["collision_approximation"] for o in assembly)),
            "collision_faces_median": int(statistics.median(collision_faces)) if collision_faces else None,
            "collision_faces_max": max(collision_faces) if collision_faces else None,
            "size_confidence": dict(Counter(o["size"]["confidence"] for o in assembly)),
            # Non-watertight meshes are the ones the demotion rule watches for.
            "non_watertight": [
                {"object_id": r["object_id"], "label": r.get("label"),
                 "body_type": next((o["physics"]["body_type"] for o in assembly
                                    if o["object_id"] == r["object_id"]), None)}
                for r in records if r.get("status") == "success"
                and not r.get("stats", {}).get("watertight", True)
            ],
        },
    }


def dualcraft_runs() -> list[dict]:
    runs = []
    for run in sorted(DUALCRAFT_RUNS.glob("*/job.json")):
        run = run.parent
        job = read_json(run / "job.json", {})
        history = {h["stage"]: h["at"] for h in job.get("history", [])}
        cost = read_json(run / "rev" / "cost.json", {}).get("total", {})
        report = read_json(run / "product" / "compile_report.json", {})
        urdf = run / "product" / "model.urdf"
        links = joints = movable = 0
        if urdf.exists():
            root = ET.parse(urdf).getroot()
            links = len(root.findall("link"))
            joints = len(root.findall("joint"))
            movable = sum(1 for j in root.findall("joint") if j.get("type") != "fixed")
        page = ROOT / "docs" / "dualcraft" / f"{run.name}.html"
        runs.append({
            "name": run.name,
            "history": history,
            "model": job.get("settings", {}).get("model"),
            "cost_usd": round(cost.get("costs_usd", {}).get("total", 0), 2),
            "tokens": cost.get("tokens", {}).get("total_tokens"),
            "turns": report.get("metrics", {}).get("turn_count"),
            "status": report.get("status"),
            "links": links, "joints": joints, "movable": movable,
            "page": f"../../docs/dualcraft/{run.name}.html" if page.exists() else None,
        })
    return runs


def worked_example(generation: Path, object_id: str, dualcraft_run: str) -> dict:
    record = next((r for r in read_json(generation / "objects.json", []) if r["object_id"] == object_id), None)
    if record is None:
        return {}
    manifest = read_json(generation / "manifest.json", {})
    handoff = Path(manifest["source_handoff"]["path"])
    asm = next((o for o in read_json(generation / "assembly_handoff" / "objects.json", [])
                if o["object_id"] == object_id), {})
    run = DUALCRAFT_RUNS / dualcraft_run
    renders = run / "rev" / "blender" / "renders"
    compare = sorted(renders.glob("compare-revision-*-ref_photo_0-*.png"))
    scan = read_json(run / "inputs" / "scannet_handoff.json", {}).get("size", {})
    estimate = next((o for o in read_json(generation / "physics_appearance.json", {}).get("objects", [])
                     if o["object_id"] == object_id), {})
    urdf_mass = None
    if (run / "product" / "model.urdf").exists():
        root = ET.parse(run / "product" / "model.urdf").getroot()
        urdf_mass = round(sum(float(m.get("value")) for m in root.iter("mass")), 1)
    glb_extent = None
    info = read_json(run / "viewer_links" / "info.json")
    if info:
        lo = [min(v["bounds"][0][k] for v in info.values() if v["bounds"]) for k in range(3)]
        hi = [max(v["bounds"][1][k] for v in info.values() if v["bounds"]) for k in range(3)]
        # Link bounds are link-local; the housing link sits at the origin and holds the
        # envelope, so its box is the asset size.
        housing = info.get("housing", {}).get("bounds")
        if housing:
            lo, hi = housing
        glb_extent = [round(hi[k] - lo[k], 3) for k in range(3)]
    return {
        "object_id": object_id,
        "label": record.get("label"),
        "seq": record["source"]["seq"],
        "boxed": image_uri(handoff / "keyframes" / object_id / f"{record['source']['seq']:04d}_boxed.jpg", 800),
        "sam3d_preview": image_uri(generation / "objects" / object_id / "primary" / "sam3d" / "preview.png", 900),
        "sam3d": {
            "faces": record.get("stats", {}).get("faces"),
            "watertight": record.get("stats", {}).get("watertight"),
            "seconds": record.get("stats", {}).get("seconds"),
            "dimensions_m": asm.get("size", {}).get("dimensions_m"),
            "size_confidence": asm.get("size", {}).get("confidence"),
            "size_flags": asm.get("size", {}).get("flags"),
            "physics": asm.get("physics"),
            "estimated_mass_kg": estimate.get("mass_kg"),
        },
        "dualcraft": {
            "run": dualcraft_run,
            "compare": image_uri(compare[0], 1100) if compare else None,
            "dimensions_m": glb_extent,
            "scan": scan,
            "urdf_mass_kg": urdf_mass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generation", type=Path, help="SAM3D generation output, e.g. output/scene0000_00/gt")
    parser.add_argument("--example", default="obj_0038", help="object shown through both routes")
    parser.add_argument("--dualcraft-run", default="scannet_s0000_obj0038_v2")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generation = args.generation.resolve()
    routing = read_json(generation / "routing.json", {})
    payload = {
        "routing": {"route_counts": routing.get("route_counts"), "overridden": routing.get("overridden", [])},
        "generation": generation_stats(generation),
        "dualcraft_runs": dualcraft_runs(),
        "example": worked_example(generation, args.example, args.dualcraft_run),
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
