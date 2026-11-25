"""Hand edits to a finished appearance-agent run, recorded, without calling the model again.

    PYTHONPATH=src .venv-sam3d/bin/python -m structuregen.agent.amend <agent run dir> \\
        --paint <segment> <face> <r,g,b> [--paint ...] --reason "<why>"

The agent's own model is kept once as appearance_agent.blend; appearance.blend becomes the edited
one, so export and the page pick the edit up. Every call appends to amendments.json (what, from
which colour, to which, why, when). Paints use the sg.paint operator with item "manual".
Re-renders final_compare/ afterwards.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .. import config

SCRIPT = Path(__file__).parent / "blender" / "blender_amend.py"
DUMP = Path(__file__).parents[1] / "export" / "blender_dump.py"


def face_colours(blend: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dump.json"
        subprocess.run([str(config.BLENDER), "-b", str(blend), "--python", str(DUMP), "--", str(out)],
                       capture_output=True, text=True, check=True)
        doc = json.loads(out.read_text())
    return {s["name"]: {"rgb": s["rgb"], "item": s["item"]} for o in doc["objects"] for s in o["slots"] if s["name"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent", type=Path)
    parser.add_argument("--paint", nargs=3, action="append", required=True, metavar=("SEG", "FACE", "RGB"))
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    agent = args.agent.resolve()
    blend, original = agent / "appearance.blend", agent / "appearance_agent.blend"
    if not original.exists():
        shutil.copyfile(blend, original)
    before = face_colours(blend)
    edits = []
    for seg, face, rgb in args.paint:
        name = f"{seg}_{face}"
        if name not in before:
            raise SystemExit(f"no face {name} in {blend}")
        edits.append({"seg": seg, "face": face, "rgb": [int(c) for c in rgb.split(",")], "roughness": 0.9,
                      "was": before[name]})
    with tempfile.TemporaryDirectory() as tmp:
        e = Path(tmp) / "edits.json"
        e.write_text(json.dumps(edits))
        new = Path(tmp) / "appearance.blend"
        r = subprocess.run([str(config.BLENDER), "-b", str(blend), "--python", str(SCRIPT), "--", str(e), str(new)],
                           capture_output=True, text=True)
        if r.returncode or not new.is_file():
            raise SystemExit(f"Blender failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        shutil.copyfile(new, blend)
    after = face_colours(blend)
    for ed in edits:
        got = after[f"{ed['seg']}_{ed['face']}"]
        if got["rgb"] != ed["rgb"] or got["item"] != "manual":
            raise SystemExit(f"{ed['seg']}_{ed['face']} reads back as {got}")
    log = agent / "amendments.json"
    entries = json.loads(log.read_text()) if log.exists() else []
    entries.append({"at": datetime.now().isoformat(timespec="seconds"), "reason": args.reason,
                    "paint": [{k: v for k, v in ed.items() if k != "roughness"} for ed in edits]})
    log.write_text(json.dumps(entries, indent=1, ensure_ascii=False))
    from .run import final_compare
    from ..export.handoff import white_model
    final_compare(white_model(agent), agent)
    for ed in edits:
        print(f"AMENDED {ed['seg']}_{ed['face']}: {ed['was']['rgb']} ({ed['was']['item']}) -> {ed['rgb']} (manual)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
