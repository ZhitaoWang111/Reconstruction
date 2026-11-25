"""Runs inside Blender for a whole appearance session: one JSON command per stdin line.

    blender -b <walls.blend> --python src/structuregen/agent/blender/blender_worker.py -- <work dir>

Answers each command with one stdout line `SG_REPLY <json>`. Commands:

    {"cmd": "edit", "code": ..., "segment": ...}   snapshot -> run code -> check; a code error or a
                                                    hard problem restores the snapshot
    {"cmd": "check", "segment": ...}                checks on the scene as it is (finish_segment)
    {"cmd": "progress", "segment": ..., "items": [...], "faces": [...]}
    {"cmd": "passes", "cams": [...], "out": ..., "scale": ...}   compare passes
    {"cmd": "save", "path": ...}
    {"cmd": "quit"}

The sg operators and the checks are exec'd from their source files into this process, so agent
code reaches them only as `bpy.ops.sg.*`. Agent code runs with `bpy` in its namespace, as the
mcp-for-blender addon's execute_code does.
"""
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
WORK = Path(sys.argv[sys.argv.index("--") + 1])
WORK.mkdir(parents=True, exist_ok=True)
SNAPSHOT = WORK / "pre_edit.blend"
bpy.context.preferences.filepaths.save_version = 0      # no .blend1 next to every snapshot

ns = {"__name__": "sg_runtime"}
for name in ("sg_ops.py", "checks.py", "blender_compare.py"):
    exec(compile((HERE / name).read_text(), str(HERE / name), "exec"), ns)


def reload_ops():
    """Operators are registered per process, but open_mainfile resets the scene: re-register."""
    ns["register"]()


REF = ns["reference"]()


def reply(payload):
    sys.stdout.write("SG_REPLY " + json.dumps(payload) + "\n")
    sys.stdout.flush()


def agent_error(exc):
    """One line an agent can act on: where in its code, and the innermost message.

    Operator failures arrive wrapped ("RuntimeError: Error: Python: Traceback ... ValueError: ...");
    the line that matters is the last 'Name: message' line, not our file paths.
    """
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<agent>":
            line = frame.lineno
    message = f"{type(exc).__name__}: {exc}"
    inner = [l.strip() for l in str(exc).splitlines() if l.strip() and not l.strip().startswith(("File ", "Location:", "Traceback", "raise ", "^"))]
    named = [l for l in inner if ": " in l and l.split(": ", 1)[0].replace(".", "").isidentifier()
             and not l.startswith(("Error: Python", "RuntimeError: Error"))]
    if named:
        message = named[-1]
    return (f"line {line}: " if line else "") + message


def edit(cmd):
    segment = cmd["segment"]
    before = ns["snapshot"]()
    bpy.ops.wm.save_as_mainfile(filepath=str(SNAPSHOT), copy=True)
    out = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(cmd["code"], "<agent>", "exec"), {"bpy": bpy})
    except Exception as exc:
        error = agent_error(exc)
    result = None
    if error is None:
        result = ns["check"](REF, before, segment, [tuple(k) for k in cmd.get("keep", [])])
    rolled_back = False
    if error is not None or result["hard"]:
        bpy.ops.wm.open_mainfile(filepath=str(SNAPSHOT))
        reload_ops()
        rolled_back = True
    if result is None or rolled_back:
        after = ns["check"](REF, None, segment)       # report the trims of the restored scene
        trims = after["trims"]
        state = after["state"]
    else:
        trims, state = result["trims"], result["state"]
    changed = sorted(n for n in state if state[n] != before.get(n)) if not rolled_back else []
    return {"ok": not rolled_back, "rolled_back": rolled_back, "error": error,
            "hard": (result or {}).get("hard", []), "trims": trims,
            "stdout": "\n".join(l for l in out.getvalue().splitlines() if l.startswith("SG "))[-4000:],
            "changed": changed}


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        cmd = json.loads(line)
        kind = cmd["cmd"]
        if kind == "edit":
            reply(edit(cmd))
        elif kind == "check":
            r = ns["check"](REF, None, cmd["segment"])
            reply({"hard": r["hard"], "trims": r["trims"]})
        elif kind == "progress":
            reply(ns["progress"](cmd["segment"], cmd.get("items", []), cmd.get("faces", ["front"])))
        elif kind == "passes":
            ns["render_passes"](cmd["out"], float(cmd.get("scale", 0.5)), cmd["cams"])
            reply({"ok": True})
        elif kind == "save":
            bpy.ops.wm.save_as_mainfile(filepath=cmd["path"], copy=True)
            reply({"ok": True})
        elif kind == "quit":
            reply({"ok": True})
            break
        else:
            reply({"error": f"unknown command {kind!r}"})
    except Exception as exc:
        reply({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]})
