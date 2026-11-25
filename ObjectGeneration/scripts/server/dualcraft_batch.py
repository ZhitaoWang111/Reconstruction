"""Run a batch of DualCraft photo-route jobs on this server, one after another.

Runs on the DualCraft server, inside its repository (``PYTHONPATH=src``), started detached
so an SSH drop does not stop it::

    cd /mnt/data/usr/wzt/submodules/dualcraft/dual-craft
    DISPLAY=:99 PYTHONPATH=src setsid nohup .venv/bin/python <this file> <batch dir> \\
        --order obj_0038,obj_0033 --max-cost-usd 20 > <batch dir>/batch.log 2>&1 < /dev/null &

``<batch dir>`` holds one directory per object as ``prepare_dualcraft_inputs.py`` wrote
them (``photo_<k>.png``; ``conversion.json`` is ignored here).  For each object in
``--order`` the script creates a DualCraft job from the photos alone (``create_job``, no
``video``: DualCraft anchors the cameras and estimates the size itself) and runs it with
the job driver in this process, then moves to the next.  DualCraft allows one job at a
time per output root, so jobs never overlap.

State goes to ``<batch dir>/batch.json`` after every step: which job each object got,
its stage, cost and turns.  A rerun skips objects whose job finished (``done``) and
starts a fresh job for the rest, so a batch can be resumed after a crash.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def job_summary(job: Path) -> dict:
    data = json.loads((job / "job.json").read_text())
    cost = None
    for path in sorted(job.glob("articraft_data/records/*/revisions/*/cost.json")):
        total = json.loads(path.read_text()).get("total", {})
        cost = round(total.get("costs_usd", {}).get("total", 0.0), 4)
    report = job / "product" / "compile_report.json"
    turns = json.loads(report.read_text()).get("metrics", {}).get("turn_count") if report.is_file() else None
    return {"stage": data.get("stage"), "error": data.get("error"), "category": data.get("category"),
            "cost_usd": cost, "turns": turns, "relay_mode": data.get("relay_mode"),
            "history": data.get("history")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", type=Path, help="directory with one subdirectory of photos per object")
    parser.add_argument("--order", required=True, help="comma-separated object ids, run in this order")
    parser.add_argument("--prefix", default="scannet_s0000", help="job name prefix")
    parser.add_argument("--max-cost-usd", type=float, default=20.0, help="per-job cost cap")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--geometry-turn-cap", type=int, default=8,
                        help="blender_edit calls allowed before DualCraft leaves its geometry phase")
    parser.add_argument("--dry-run", action="store_true",
                        help="create the jobs and stop them at 'prepared' (no model call); state in batch.dryrun.json")
    args = parser.parse_args()

    from demo import jobs

    batch = args.batch.resolve()
    state_path = batch / ("batch.dryrun.json" if args.dry_run else "batch.json")
    if args.dry_run:
        args.prefix = "dryrun_" + args.prefix
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"objects": {}}
    state.update(order=args.order.split(","), max_cost_usd=args.max_cost_usd, started_at=state.get("started_at") or now())
    write_state(state_path, state)

    for oid in state["order"]:
        entry = state["objects"].setdefault(oid, {})
        if entry.get("stage") == "done":
            print(f"{oid}: done before ({entry.get('job')}), skipped", flush=True)
            continue
        photos = sorted((batch / oid).glob("photo_*.png"), key=lambda p: int(p.stem.split("_")[1]))
        if not photos:
            entry.update(stage="failed", error="no photos in the batch directory")
            write_state(state_path, state)
            continue
        attempt = int(entry.get("attempt", 0)) + 1
        name = f"{args.prefix}_{oid.replace('_', '')}_photos{'' if attempt == 1 else f'_r{attempt}'}"
        settings = {"max_cost_usd": args.max_cost_usd, "max_turns": args.max_turns,
                    "geometry_turn_cap": args.geometry_turn_cap}
        entry.update(job=name, attempt=attempt, photos=[p.name for p in photos], started_at=now(),
                     stage="created", error=None, settings=settings)
        write_state(state_path, state)
        print(f"{oid}: job {name} with {len(photos)} photo(s), {settings}", flush=True)
        started = time.time()
        try:
            job = jobs.create_job([(p.name, p.read_bytes()) for p in photos], name=name, dry_run=args.dry_run,
                                  settings=settings)
            code = jobs.run_job(job)
            entry.update(exit=code, **job_summary(job))
        except Exception as exc:
            traceback.print_exc()
            entry.update(stage="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
        entry.update(seconds=round(time.time() - started), finished_at=now())
        write_state(state_path, state)
        print(f"{oid}: {entry.get('stage')} in {entry['seconds']}s, cost {entry.get('cost_usd')}, "
              f"turns {entry.get('turns')}, {entry.get('error') or ''}", flush=True)
    state["finished_at"] = now()
    write_state(state_path, state)
    print("BATCH_DONE", json.dumps({o: e.get("stage") for o, e in state["objects"].items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
