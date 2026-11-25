"""Register a DualCraft run as the asset of one handoff object.

DualCraft runs elsewhere (the DSW server) and leaves a run directory with the
product (URDF + per-link OBJ/MTL/textures + object.usdc), its inputs (reference
photos, size.json, blueprint) and its trajectory.  This copies what the export
chain needs into the generation directory, next to where SAM3D would have put
its result:

    <generation>/objects/<object_id>/<view>/dualcraft/
        product/            model.urdf, object.usdc, assets/meshes/...  (verbatim)
        inputs/             prompt.json, size.json, blueprint/, scannet_handoff.json
        trajectory.jsonl[.zst]
        report.json         what was imported, hashes, the run's own numbers

and records the object in ``dualcraft.json``.  ``handoff_io.write_index`` reads
that registry when it rebuilds ``objects.json``, so the record survives a SAM3D
rerun and points at the DualCraft asset with ``generator: dualcraft``.

The run directory is either the DualCraft demo job (``output/demo/<run>``, with
``product/``) or a local copy laid out the same way.  ``scannet_handoff.json``
(written when the inputs were converted from the handoff) must carry
``object_frame``: where the DualCraft object frame sits in the scan.

Usage::

    PYTHONPATH=src .venv-sam3d/bin/python -m objgen.dualcraft.import_dualcraft output/scene0000_00/gt \\
        obj_0038 /path/to/dualcraft/run [--view primary]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from ..config import ROOT

from . import dualcraft_product as dp  # noqa: E402
from .. import handoff_io as hio  # noqa: E402

STAGE = "dualcraft"
REGISTRY = "dualcraft.json"


def find_trajectory(run: Path) -> Path | None:
    for pattern in ("rev/traces/trajectory.jsonl", "rev/traces/trajectory.jsonl.zst",
                    "**/traces/trajectory.jsonl.zst", "**/traces/trajectory.jsonl"):
        hits = sorted(run.glob(pattern))
        if hits:
            return hits[0]
    return None


def find_inputs(run: Path) -> Path:
    for candidate in (run / "inputs", run / "product" / "inputs"):
        if (candidate / "prompt.json").is_file():
            return candidate
    raise FileNotFoundError(f"{run}: no inputs/prompt.json")


def import_run(generation: Path, object_id: str, run: Path, view: str = "primary") -> dict:
    generation, run = Path(generation).resolve(), Path(run).resolve()
    manifest = json.loads((generation / "manifest.json").read_text())
    handoff = Path(manifest["source_handoff"]["path"])
    scene = {record["object_id"]: record for record in json.loads((handoff / "objects.json").read_text())}
    if object_id not in scene:
        raise SystemExit(f"{object_id} is not in {handoff}")

    product_src = run / "product"
    if not (product_src / "model.urdf").is_file():
        raise SystemExit(f"{product_src}/model.urdf not found")
    product = dp.read_product(product_src)        # fails early on a broken URDF or missing OBJ
    missing = [str(path) for link in product.links.values() for path, _ in link.visuals if not path.is_file()]
    if missing:
        raise SystemExit("product is incomplete, missing meshes: " + ", ".join(missing[:5]))
    inputs_src = find_inputs(run)
    conversion = inputs_src / "scannet_handoff.json"
    if not conversion.is_file():
        raise SystemExit(f"{conversion} not found: the export needs the object frame recorded "
                         "when the handoff was converted to DualCraft inputs")
    frame = json.loads(conversion.read_text()).get("object_frame")
    if not frame or not {"origin_xy", "bottom_z", "front", "left"} <= set(frame):
        raise SystemExit(f"{conversion} lacks object_frame (origin_xy, bottom_z, front, left)")

    stage = hio.stage_dir(generation, object_id, view).parent / STAGE
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copytree(product_src, stage / "product")
    shutil.copytree(inputs_src, stage / "inputs")
    trajectory_src = find_trajectory(run)
    trajectory = None
    if trajectory_src is not None:
        trajectory = stage / trajectory_src.name
        shutil.copyfile(trajectory_src, trajectory)

    job = json.loads((run / "job.json").read_text()) if (run / "job.json").is_file() else {}
    compile_report = json.loads((product_src / "compile_report.json").read_text()) \
        if (product_src / "compile_report.json").is_file() else {}
    cost = next((json.loads(p.read_text()).get("total") for p in
                 (run / "rev" / "cost.json", *sorted(run.glob("**/revisions/*/cost.json"))) if p.is_file()), None)
    movable = [j.name for j in product.joints if j.type != "fixed"]
    report = {
        "object_id": object_id,
        "label": scene[object_id].get("label"),
        "view": view,
        "status": "success",
        "generator": STAGE,
        "source": {"run": str(run), "job": job.get("name") or run.name,
                   "model": (job.get("settings") or {}).get("model"),
                   "history": job.get("history"),
                   "compile_status": compile_report.get("status"),
                   "turns": (compile_report.get("metrics") or {}).get("turn_count"),
                   "cost_usd": round(cost["costs_usd"]["total"], 4) if cost else None},
        "frame": dp.ROOT_FRAME,
        "object_frame": frame,
        "links": len(product.links),
        "joints": len(product.joints),
        "movable_joints": movable,
        "outputs": {str(path.relative_to(stage)): hio.sha256(path)
                    for path in sorted((stage / "product").glob("*")) if path.is_file()},
        "trajectory": trajectory.name if trajectory else None,
        "finished_at": hio.now(),
    }
    hio.atomic_json(stage / "report.json", report)

    registry_path = generation / REGISTRY
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    registry[object_id] = {"view": view, "stage": str(stage.relative_to(generation)), "run": str(run),
                           "imported_at": report["finished_at"]}
    hio.atomic_json(registry_path, dict(sorted(registry.items())))
    hio.reindex(generation)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generation", type=Path, help="Generation directory (has manifest.json)")
    parser.add_argument("object_id")
    parser.add_argument("run", type=Path, help="DualCraft run directory (has product/model.urdf)")
    parser.add_argument("--view", default="primary", choices=hio.ROLES)
    args = parser.parse_args(argv)
    report = import_run(args.generation, args.object_id, args.run, args.view)
    print(f"IMPORTED {report['object_id']} {report['label']}: {report['links']} links, "
          f"movable {report['movable_joints']} -> objects/{args.object_id}/{args.view}/{STAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
