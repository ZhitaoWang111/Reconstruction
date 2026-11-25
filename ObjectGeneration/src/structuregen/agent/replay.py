"""Replay a finished appearance-agent run onto a rebuilt white model, without the model.

    PYTHONPATH=src .venv-sam3d/bin/python -m structuregen.agent.replay <agent run dir> <new white-model run dir> [--stop-before wrapup]

For a geometry fix that leaves every segment's local frame alone (the agent's paints and trims
are in segment-local metres): every declare, edit and finish of the transcript is sent again,
in order, through a fresh Session on the new run, so the same gates and checks run. Compare
calls are skipped. Writes ``<new run>/agent_<stamp>_replay/`` with appearance.blend, a new
session transcript, report.json (the original's usage and cost, ``replayed_from`` added) and
the original usage.jsonl and calls/. Stops at the first step whose outcome differs.

``--stop-before wrapup`` replays the walls only and saves the scene there, as ``walls.blend`` in
the output, for an agent run that adds what the source run did not have (floors) and redoes the
wrap-up. Bodies the source run never saw are left for that run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from .session import Session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--stop-before", choices=["wrapup"], default=None)
    args = parser.parse_args(argv)
    src, run = args.agent.resolve(), args.run.resolve()
    report = json.loads((src / "report.json").read_text())
    steps = [json.loads(line) for line in (src / "session/transcript.jsonl").read_text().splitlines() if line.strip()]
    order = [w["wall"] for w in report["walls"] if w["wall"] in report["wall_state"]]
    out = run / f"{src.name}_replay"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    s = Session(run, out / "session", order=order)
    replayed = 0
    try:
        for st in steps:
            tool, a, before = st["tool"], st["args"], st["result"]
            if tool in ("brief", "compare"):
                continue
            if args.stop_before == "wrapup" and st["segment"] in ("wrapup", "committed"):
                break
            if tool == "declare":
                r = s.declare(a["n"], a["findings"])
            elif tool == "execute_blender_code":
                r = s.execute_blender_code(a["code"], a.get("description", ""))
            elif tool == "finish_segment":
                r = s.finish_segment(a["n"], a["summary"], a.get("dropped"))
            elif tool == "commit":
                r = s.commit()
            else:
                raise SystemExit(f"step {st['n']}: unknown tool {tool}")
            same = (bool(r.get("refused")) == bool(before.get("refused"))
                    and bool(r.get("rolled_back")) == bool(before.get("rolled_back"))
                    and r.get("trims", []) == before.get("trims", []))
            if not same:
                raise SystemExit(f"step {st['n']} {tool} came out differently:\nbefore: {before.get('text')}\nnow: {r.get('text')}")
            replayed += 1
        if args.stop_before == "wrapup":
            s.save(out / "walls.blend")
            (out / "walls_done.json").write_text(json.dumps(s.walls, indent=1, ensure_ascii=False, default=str))
            (out / "report.json").write_text(json.dumps({"replayed_from": str(src), "stopped_before": "wrapup",
                                                          "walls": list(s.walls)}, indent=1))
            print(f"REPLAYED {replayed} wall steps -> {out / 'walls.blend'}")
            return 0
        if s.current != "committed":
            raise SystemExit(f"replay ended in {s.current!r}, not committed")
        s.save(out / "appearance.blend")
    finally:
        s.close()
    new = {**report, "wall_state": s.walls, "replayed_from": str(src), "tool_calls": s.calls,
           "replay_note": "same model replies and cost as the source run; edits re-applied to a rebuilt white model"}
    (out / "report.json").write_text(json.dumps(new, indent=1, ensure_ascii=False, default=str))
    shutil.copyfile(src / "usage.jsonl", out / "usage.jsonl")
    shutil.copytree(src / "calls", out / "calls")
    print(f"REPLAYED {replayed} steps -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
