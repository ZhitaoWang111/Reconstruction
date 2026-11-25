"""Summarise fetched DualCraft jobs for review: what DualCraft built, how it compares with
the scan, and whether its joints are sound.

For each job directory (as ``fetch_dualcraft_batch.py`` leaves them under
``output/dualcraft/``) this prints:

* the run: stage, cost, turns, minutes;
* the blueprint: what DualCraft took the object to be, its estimated size and mass, parts;
* the cameras DualCraft's own model guessed for the photos (photo route);
* the URDF's movable joints and whether the agent confirmed each axis;
* the closed asset's size against the scan box the photos were cut for (``conversion.json``
  from ``prepare_dualcraft_inputs.py``): footprint sides are compared long-to-long and
  short-to-short, since the asset's front is DualCraft's guess;
* our geometry checks (``dualcraft_product.interference``): bodies overlapping at rest, a
  door that swings into its cabinet or inward;
* diff-ledger items left open, refused finish attempts and expired rework windows.

Nothing is written.

Usage::

    PYTHONPATH=src .venv-sam3d/bin/python -m objgen.tools.review_dualcraft_jobs output/dualcraft/<job> [...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ..config import ROOT

from ..dualcraft import dualcraft_product as dp  # noqa: E402

DEFAULT_INPUTS = ROOT / "output" / "scene0000_00" / "gt" / "dualcraft_inputs"


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def trajectory_lines(run: Path) -> list[dict]:
    paths = sorted((run / "rev" / "traces").glob("trajectory.jsonl*"))
    if not paths:
        return []
    path = paths[0]
    text = (subprocess.run(["zstd", "-dc", str(path)], check=True, capture_output=True).stdout.decode()
            if path.suffix == ".zst" else path.read_text())
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def ledger(run: Path) -> list[dict]:
    """Final state of every diff-ledger item."""
    items: dict[str, dict] = {}
    path = run / "rev" / "blender" / "diff_ledger.jsonl"
    if not path.is_file():
        return []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        item = event.get("item")
        if not item:
            continue
        entry = items.setdefault(item["id"], {"id": item["id"], "kind": item.get("kind"),
                                              "parts": item.get("parts"), "note": item.get("note")})
        if event["event"] in ("open", "reopen"):
            entry["status"] = "open"
        elif event["event"] == "resolve":
            entry["status"] = "resolved"
        elif event["event"] == "drop":
            entry["status"] = "dropped"
    return list(items.values())


def review(run: Path, inputs_root: Path) -> dict:
    fetched = read_json(run / "fetched.json", {})
    oid = fetched.get("object_id")
    entry = fetched.get("batch_entry", {})
    job = read_json(run / "job.json", {})
    history = {h["stage"]: h["at"] for h in job.get("history", [])}
    cost = (read_json(run / "rev" / "cost.json", {}) or {}).get("total", {}).get("costs_usd", {}).get("total")
    out = {"object_id": oid, "job": run.name, "stage": job.get("stage"), "error": job.get("error"),
           "cost_usd": entry.get("cost_usd") if entry.get("cost_usd") is not None else
           (round(cost, 4) if cost is not None else None),
           "turns": entry.get("turns"),
           "minutes": round(entry["seconds"] / 60, 1) if entry.get("seconds") else None}

    prompt = read_json(run / "inputs" / "prompt.json", {})
    estimates = (read_json(run / "inputs" / "blueprint" / "source.json", {}) or {}).get("estimates", {})
    parts = prompt.get("parts", [])
    out["blueprint"] = {"label": estimates.get("label"), "size_m": estimates.get("size_m"),
                        "mass_kg": estimates.get("mass_kg"), "parts": len(parts),
                        "movable": [p["id"] for p in parts if p.get("movable")],
                        "size_assumption": estimates.get("size_assumption")}
    manifest = read_json(run / "inputs" / "reference_manifest.json", {})
    out["anchors"] = [{"photo": image["id"], **{k: (image.get("anchor") or {}).get(k)
                                                 for k in ("azimuth_deg", "elevation_deg", "confidence")}}
                      for image in manifest.get("images", [])]

    report = read_json(run / "product" / "compile_report.json", {})
    out["compile"] = {"status": report.get("status"),
                      "warnings": [w["message"].split("\n")[0][:160] for w in report.get("warnings", [])]}
    events = trajectory_lines(run)
    out["finish_attempts"] = [e.get("accepted") for e in events if e.get("type") == "harness_finish_attempt"]
    out["rework_expired"] = sum(1 for e in events if e.get("type") == "diff_loop_rework_expired")
    out["ledger_open"] = [f"{i['id']} {i['kind']}: {(i['note'] or '')[:100]}" for i in ledger(run)
                          if i.get("status") == "open"]

    if not (run / "product" / "model.urdf").is_file():
        out["geometry"] = "no product"
        return out
    product = dp.read_product(run / "product")
    axes = {}
    for event in events:
        if event.get("type") == "joint_axis_set":
            for name, joint in (event.get("joints") or {}).items():
                axes[name] = joint
    out["joints"] = [{"name": j.name, "type": j.type, "axis": [round(float(v), 3) for v in j.axis],
                      "range": [j.lower, j.upper],
                      "axis_source": (axes.get(j.name) or {}).get("source", "placeholder (never set)"),
                      "changed": (axes.get(j.name) or {}).get("changed")}
                     for j in product.joints if j.type != "fixed"]
    materials = dp.link_materials(product, dp.read_blueprint(run))
    bodies = dp.build_bodies(product, materials)
    out["checks"] = {k: v for k, v in dp.interference(product, bodies).items() if not k.endswith("_m")}
    out["bodies"] = {name: len(body.links) for name, body in bodies.items()}
    bounds = dp.rest_bounds(product)
    extent = bounds[1] - bounds[0]                  # x depth, y width, z height
    out["asset_m"] = {"depth": round(float(extent[0]), 3), "width": round(float(extent[1]), 3),
                      "height": round(float(extent[2]), 3)}
    conversion = read_json(inputs_root / str(oid) / "conversion.json") if oid else None
    if conversion:
        box = conversion["scan_box"]
        long_, short = sorted((float(extent[0]), float(extent[1])), reverse=True)
        out["scan_m"] = {"long": box["long"], "short": box["short"], "height": box["height"]}
        out["asset_over_scan"] = {"height": round(float(extent[2]) / box["height"], 3),
                                  "long": round(long_ / box["long"], 3),
                                  "short": round(short / box["short"], 3),
                                  "width_is_long_side": bool(extent[1] >= extent[0])}
    renders = run / "rev" / "blender" / "renders"
    compares = sorted(renders.glob("compare-revision-*.png"))
    if compares:
        last = compares[-1].name.split("-ref_")[0]
        out["compare_renders"] = [str(p) for p in compares if p.name.startswith(last)]
    return out


def show(r: dict) -> None:
    print(f"== {r['object_id']} {r['job']}: {r['stage']}, ${r['cost_usd']}, {r['turns']} turns, "
          f"{r['minutes']} min" + (f", error: {r['error']}" if r.get("error") else ""))
    b = r["blueprint"]
    size = b.get("size_m") or {}
    print(f"   blueprint: {b['label']} | est. {size.get('width')} w x {size.get('depth')} d x "
          f"{size.get('height')} h m, {b['mass_kg']} kg | {b['parts']} parts, movable {b['movable']}")
    print("   anchors: " + "; ".join(f"{a['photo']} az {a['azimuth_deg']} el {a['elevation_deg']} "
                                    f"conf {a['confidence']}" for a in r["anchors"]))
    print(f"   compile {r['compile']['status']}, warnings {len(r['compile']['warnings'])}"
          + "".join(f"\n     - {w}" for w in r["compile"]["warnings"]))
    print(f"   finish attempts {r['finish_attempts']}, rework windows expired {r['rework_expired']}, "
          f"ledger open {len(r['ledger_open'])}" + "".join(f"\n     - {x}" for x in r["ledger_open"]))
    if r.get("geometry") == "no product":
        print("   no product to check")
        return
    print("   joints: " + "; ".join(f"{j['name']} {j['type']} {j['axis']} [{j['axis_source']}"
                                   f"{', flipped' if j['changed'] else ''}]" for j in r["joints"]))
    print(f"   bodies {r['bodies']}")
    checks = r["checks"]
    print(f"   checks: rest overlaps {checks['rest_overlaps']}, swept into parent "
          f"{[s['joint'] for s in checks['swept_into_parent']]}, inward {[s['joint'] for s in checks['moves_inward']]}")
    a = r["asset_m"]
    print(f"   asset {a['width']} w x {a['depth']} d x {a['height']} h m", end="")
    if r.get("scan_m"):
        s, q = r["scan_m"], r["asset_over_scan"]
        print(f" | scan {s['long']} x {s['short']} x {s['height']} m | asset/scan: height {q['height']}, "
              f"long {q['long']}, short {q['short']}" + ("" if q["width_is_long_side"] else
                                                         " (asset width is its SHORT side)"))
    else:
        print()
    for path in r.get("compare_renders", []):
        print(f"   compare: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="fetched job directories")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS,
                        help="dualcraft_inputs directory with <object_id>/conversion.json")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the text summary")
    args = parser.parse_args()
    results = [review(run.resolve(), args.inputs) for run in args.runs]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for result in results:
            show(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
