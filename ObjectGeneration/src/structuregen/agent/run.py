"""The appearance agent: gpt-6-astra over the vibeg gateway, one conversation per wall.

    PYTHONPATH=src .venv-sam3d/bin/python -m structuregen.agent.run [run dir] [--fake] [--max-usd 20]

Each wall starts a fresh conversation holding the task card, the plan view, the summaries of the
walls already done, and this wall's photos plus a compare picture of the current model. The model
then calls tools until finish_segment passes or the wall's reply budget runs out; after the last
wall a wrap-up conversation paints the faces no photo shows and commits.

Limits: MAX_REPLIES model replies per wall (and for the wrap-up), MAX_IDLE replies in a row
without a tool call end the wall, and the whole run stops before a call that could cross max_usd.
A wall that runs out is marked unconverged: the edits that passed the checks stay, its open trim
problems are removed, and unpainted room-side faces get a neutral colour marked as such.

`--fake` replaces the model with scripted replies (no network, no cost) to test the loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .. import config
from .session import Session

MAX_REPLIES = 15
MAX_IDLE = 3
NEUTRAL = (200, 200, 200)

INSTRUCTIONS = """You give an empty room's floors and walls their appearance in Blender, one at a time, from photos:
first every floor (one per room), then every wall.

The walls already exist as a white model built from a scan; their shape is fixed and must not change.
For the wall you are given, decide from its photos what its room-side face looks like: paint colour
and finish, and the trims on it - skirting at the floor, a horizontal rail, and a casing around a
handoff opening (only openings listed for the wall get a casing). Doors, door leaves, windows and
window casings are made elsewhere: where the photos show a closed door or a window the wall stays
solid here, and you add nothing for them. Furniture, pictures and curtains are not part of the wall.

Work in this order for each wall:
1. Look: the original photos, the boxed photos (the wall's visible part outlined), and compare
   pictures (photo on the left, the current model from the same camera on the right).
2. declare(n, findings): write down what you see and which key frames show it. Every item you
   declare (the surface, each trim) becomes a to-do item t1, t2, ... that you then make.
3. execute_blender_code: make the items with the sg operators. Each call is checked automatically and
   answered with the problems found and a fresh compare picture; an error or a forbidden change is
   rolled back in full.
4. Compare and correct until the model matches the photos, then finish_segment(n, summary).

The model is rendered flat: a colour shows exactly as set, slightly darker only in corners, with thin
outlines at edges. The photos carry lighting and white balance; judge the paint, not the light.
Colours are sRGB 0-255.

Blender code runs in a sandbox: start with `import bpy`, use only bpy.ops.sg.* to edit, no other modules.
    bpy.ops.sg.paint(seg="<segment>", face="front", rgb=(r, g, b), roughness=0.9, item="t1")
    bpy.ops.sg.strip(seg="<segment>", z0=0.0, z1=0.10, depth=0.015, rgb=(r, g, b), kind="skirting", item="t2")
        kind is "skirting" or "rail"; z0/z1 are heights above the floor in metres; depth is how far it
        stands out from the wall (0.003-0.05 m). a0/a1 (metres along the segment from its start) are
        optional: left out, the strip runs the whole segment and stops by itself at walls that meet
        it and at openings.
    bpy.ops.sg.frame(seg="<segment>", width=0.06, depth=0.015, sides="FRONT", rgb=(r, g, b), item="t3")
        a casing around the segment's handoff opening; sides "FRONT", "BACK" or "BOTH".
    bpy.ops.sg.tiles(seg="<floor>", face="top", size=(0.30, 0.30), joint=0.004, rgb=(r, g, b), joint_rgb=(r, g, b),
                     angle=0.0, roughness=0.6, item="t1")
        a repeating pattern of tiles or planks on a face: size is one tile (length, width) in metres,
        joint the grout / gap width (0-0.02 m), rgb the tile, joint_rgb the joint, angle the rotation of
        the grid in plan (degrees about +Z; measure it against the walls in the photos). Planks: a long,
        narrow size. sg.paint on the same face instead gives a plain colour. Choose what the photos show.
A floor (room_N, threshold_*) is a body like a wall segment: declare it with the same tool (n = its name,
trims empty), its room-side face is "top". One room has one floor finish; furniture, rugs and mats on
it are not part of the floor.
Calling an operator again with the same item replaces what that item made. Pass the item number
from declare. Face names: front (room side), back, top, bottom, end_start, end_end, reveal (the
sides of an opening).

A wall no photo shows takes its paint from the faces that bound the same space (the brief names them);
a wall continued past a partition is another room and may differ.

If an item cannot be made to match after a few tries, or the photos do not show enough, drop it in
finish_segment with a reason instead of guessing. Keep replies short; the tools carry the work.
You have a limited number of replies per wall."""

WRAPUP_INSTRUCTIONS = INSTRUCTIONS + """

All walls are done. Now paint the faces no photo shows - listed in the brief - with colours inferred
from the finished walls (the same room keeps its paint; an inner wall's two sides may differ) or a
neutral assumed colour for faces outside the room. Use item="w" for these paints. Then call commit()."""


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


RGB = {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}
FRAMES = {"type": "array", "items": {"type": "integer"}, "description": "key frames of this wall / floor that show it"}
TOOLS_WALL = [
    {"type": "function", "name": "declare",
     "description": "Write down what this wall looks like and which key frames show it, before editing. "
                    "Creates the wall's to-do items: t1 = surface, then t2, t3, ... = the trims in order.",
     "parameters": _obj({
         "n": {"type": ["integer", "string"], "description": "wall number, or the floor's name (room_1, threshold_...)"},
         "findings": _obj({
             "surface": _obj({"rgb": RGB, "material": {"type": "string"},
                              "roughness": {"type": "number"}, "basis": FRAMES,
                              "pattern": {"type": "string", "description": "plain, or tiles / planks with size, joint and angle"}},
                             ["rgb", "material", "basis"]),
             "trims": {"type": "array", "items": _obj({
                 "kind": {"type": "string", "enum": ["skirting", "rail", "casing"]},
                 "z": {"type": "array", "items": {"type": "number"}, "description": "[bottom, top] m above floor"},
                 "rgb": RGB, "basis": FRAMES, "note": {"type": "string"}}, ["kind", "rgb", "basis"])},
             "notes": {"type": "string", "description": "things seen but not made here"},
         }, ["surface", "trims"])}, ["n", "findings"])},
    {"type": "function", "name": "execute_blender_code",
     "description": "Run Blender Python that edits this wall with bpy.ops.sg.*. Answered with automatic "
                    "check results, item progress and a fresh compare picture.",
     "parameters": _obj({"code": {"type": "string"}, "description": {"type": "string"}}, ["code", "description"])},
    {"type": "function", "name": "compare",
     "description": "A picture: the key-frame photo on the left, the current model from that camera on the right.",
     "parameters": _obj({"frame": {"type": "integer"}}, ["frame"])},
    {"type": "function", "name": "finish_segment",
     "description": "Final check of this wall; on success the next wall starts in a new conversation.",
     "parameters": _obj({
         "n": {"type": ["integer", "string"]}, "summary": {"type": "string", "description": "one or two sentences"},
         "dropped": {"type": "array", "items": _obj({"item": {"type": "string"}, "reason": {"type": "string"}},
                                                    ["item", "reason"])}}, ["n", "summary"])},
]
TOOLS_WRAPUP = [TOOLS_WALL[1], TOOLS_WALL[2],
                {"type": "function", "name": "commit", "description": "Whole-scene final check.",
                 "parameters": _obj({}, [])}]


def _text(t):
    return {"type": "input_text", "text": t}


def _user(parts):
    return {"role": "user", "content": parts}


class Run:
    def __init__(self, run: Path, out: Path, fake: bool, max_usd: float, resume: Path | None = None):
        from .llm import Client, image_part
        self.image_part = image_part
        start = done = None
        if resume is not None:                    # a replay's walls: start from them, work the rest
            start, done = resume / "walls.blend", json.loads((resume / "walls_done.json").read_text())
        self.session = Session(run, out / "session", start=start, done=done)
        self.resumed_from = str(resume) if resume else None
        self.out = out
        self.fake = fake
        self.max_usd = max_usd
        self.spent = 0.0
        self.replies = 0
        self.ledger = open(out / "usage.jsonl", "w")
        self.client = None if fake else Client(out / "calls")
        self.card_text = (run / "evidence/task_card.md").read_text()
        self.plan_png = run / "evidence/plan.png"
        self.walls_log = [{"wall": k, "status": v.get("status"), "replies": 0, "replayed": True}
                          for k, v in (done or {}).items()]

    # -------------------------------------------------------------- the model
    def reply(self, instructions, items, tools, wall):
        """One model reply; real or fake. Stops the run before it can cross the budget."""
        if self.fake:
            return self._fake_reply(items, wall)
        last = max(self._last_costs[-3:], default=0.6)
        if self.spent + 1.5 * last > self.max_usd:
            raise BudgetExceeded(f"spent ${self.spent:.2f}; the next reply could cross ${self.max_usd:.2f}")
        r = self.client.respond(instructions, items, tools)
        self.spent += r["cost"]
        self.replies += 1
        self._last_costs.append(r["cost"])
        self.ledger.write(json.dumps({"n": self.replies, "wall": wall, "usage": r["usage"], "cost": round(r["cost"], 4),
                                      "spent": round(self.spent, 4), "seconds": r["seconds"], "model": r["model"],
                                      "calls": [c["name"] for c in r["calls"]]}) + "\n")
        self.ledger.flush()
        return r

    _last_costs: list = []

    # -------------------------------------------------------------- tool dispatch
    def dispatch(self, call):
        s = self.session
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError as e:
            return {"text": f"arguments are not valid JSON: {e}", "refused": True}
        name = call["name"]
        if name == "declare":
            return s.declare(args.get("n"), args.get("findings") or {})
        if name == "execute_blender_code":
            return s.execute_blender_code(args.get("code", ""), args.get("description", ""))
        if name == "compare":
            return s.compare(args.get("frame"))
        if name == "finish_segment":
            dropped = {d.get("item"): d.get("reason") for d in args.get("dropped") or []}
            return s.finish_segment(args.get("n"), args.get("summary", ""), dropped)
        if name == "commit":
            return s.commit()
        return {"text": f"unknown tool {name}", "refused": True}

    def tool_output(self, call, result, items):
        """The function_call_output, and the picture as a following user message."""
        items.append({"type": "function_call_output", "call_id": call["call_id"], "output": result["text"]})
        if result.get("image"):
            items.append(_user([_text(f"Picture for that call ({Path(result['image']).name}):"),
                                self.image_part(Path(result["image"]))]))

    # -------------------------------------------------------------- a wall
    def wall_items(self, brief):
        s = self.session
        sid = s.current
        seg = s.segments[sid]
        done = {k: w.get("summary") for k, w in s.walls.items() if w.get("status") in ("done", "unconverged")}
        parts = [_text(self.card_text), _text("Plan view (numbers are the wall numbers in the task card; room_N are floors):"),
                 self.image_part(self.plan_png, "auto")]
        if done:
            parts.append(_text("Already finished: " + json.dumps(done, ensure_ascii=False)))
        if s.is_floor(sid):
            parts.append(_text(f"Now {sid}. " + brief["text"]))
            for f in seg["frames"]:
                parts += [_text(f"Key frame {f['frame']} ({f['role']}), original:"),
                          self.image_part(s.handoff / f["rgb"]),
                          _text(f"Key frame {f['frame']}, this floor's edge drawn in yellow:"),
                          self.image_part(s.run / "evidence" / f["boxed"])]
            if brief.get("image"):
                parts += [_text("Current model, compare picture:"), self.image_part(Path(brief["image"]))]
            return [_user(parts)]
        parts.append(_text(f"Now wall #{seg['number']} {sid}. " + brief["text"]))
        for f in seg["frames"]:
            parts += [_text(f"Key frame {f['frame']} ({f['role']}), original:"),
                      self.image_part(s.handoff / f["rgb"]),
                      _text(f"Key frame {f['frame']}, the wall's visible part outlined:"),
                      self.image_part(s.handoff / f["boxed"])]
        if brief.get("image"):
            parts += [_text("Current model, compare picture:"), self.image_part(Path(brief["image"]))]
        if not seg["frames"]:
            parts.append(_text("No key frame shows this wall. " + s.neighbour_note(sid) + " They are finished "
                               "before this wall; copy their paint, add no trim the photos cannot show, and "
                               "declare it with an empty basis list."))
        return [_user(parts)]

    def run_wall(self):
        s = self.session
        sid = s.current
        brief = s.opening_brief()
        items = self.wall_items(brief)
        idle, used = 0, 0
        while used < MAX_REPLIES:
            r = self.reply(INSTRUCTIONS, items, TOOLS_WALL, sid)
            used += 1
            items.extend(r["output"])
            if not r["calls"]:
                idle += 1
                if idle >= MAX_IDLE:
                    break
                items.append(_user([_text("Use a tool: declare, execute_blender_code, compare or finish_segment.")]))
                continue
            idle = 0
            moved_on = False
            for call in r["calls"]:
                result = self.dispatch(call)
                self.tool_output(call, result, items)
                if call["name"] == "finish_segment" and not result.get("refused"):
                    moved_on = True
            left = MAX_REPLIES - used
            if moved_on:
                self.walls_log.append({"wall": sid, "status": "done", "replies": used})
                return
            if left <= 3:
                items.append(_user([_text(f"{left} replies left for this wall. Finish it: fix what is open "
                                          f"or drop it with a reason, then finish_segment.")]))
        self.give_up(sid, used)

    def give_up(self, sid, used):
        """Out of replies: keep what passed, mark the wall unconverged, move on."""
        s = self.session
        wall = s.walls[sid]
        fix = []
        check = s._worker(cmd="check", segment=sid)
        for problem in check["trims"]:
            obj = problem.split(":")[0]
            fix.append(f"o = bpy.data.objects.get({obj!r})\nif o: bpy.data.objects.remove(o)")
        p = s._progress(sid)
        for face in p["unpainted"]:
            fix.append(f"bpy.ops.sg.paint(seg={sid!r}, face={face!r}, rgb={NEUTRAL}, item='unconverged')")
        if fix:
            r = s._worker(cmd="edit", code="import bpy\n" + "\n".join(fix), segment=sid)
            if r.get("rolled_back"):
                raise RuntimeError(f"could not settle {sid}: {r}")
        for item in wall["items"].values():
            if item["status"] == "open":
                item["status"], item["reason"] = "unconverged", "reply budget used up"
        wall.update(status="unconverged", summary=f"unconverged after {used} replies")
        nxt = next((x for x in s.order if x not in s.walls), None)
        if nxt:
            s.start_segment(nxt)
        else:
            s.current = "wrapup"
        self.walls_log.append({"wall": sid, "status": "unconverged", "replies": used})

    def run_wrapup(self):
        s = self.session
        brief = s.wrapup_brief()
        items = [_user([_text(self.card_text), _text("Plan view:"), self.image_part(self.plan_png, "auto"),
                        _text(brief["text"])])]
        idle, used = 0, 0
        while used < MAX_REPLIES:
            r = self.reply(WRAPUP_INSTRUCTIONS, items, TOOLS_WRAPUP, "wrapup")
            used += 1
            items.extend(r["output"])
            if not r["calls"]:
                idle += 1
                if idle >= MAX_IDLE:
                    break
                items.append(_user([_text("Use a tool: execute_blender_code, compare or commit.")]))
                continue
            idle = 0
            for call in r["calls"]:
                result = self.dispatch(call)
                self.tool_output(call, result, items)
                if call["name"] == "commit" and not result.get("refused"):
                    self.walls_log.append({"wall": "wrapup", "status": "committed", "replies": used})
                    return True
        left = []
        for sid, faces in s.wrapup_faces().items():
            p = s._worker(cmd="progress", segment=sid, items=[], faces=faces)
            left += [f"bpy.ops.sg.paint(seg={sid!r}, face={f!r}, rgb={NEUTRAL}, item='unconverged')" for f in p["unpainted"]]
        if left:
            s._worker(cmd="edit", code="import bpy\n" + "\n".join(left), segment=list(s.segments))
        self.walls_log.append({"wall": "wrapup", "status": "unconverged", "replies": used})
        return False

    # -------------------------------------------------------------- whole run
    def run(self):
        started = time.time()
        status = "ok"
        try:
            while self.session.current in self.session.segments:
                self.run_wall()
            self.run_wrapup()
        except BudgetExceeded as e:
            status = f"stopped: {e}"
        finally:
            self.session.save(self.out / "appearance.blend")
            self.session.close()
            self.ledger.close()
            report = {"status": status, "white_model": str(self.session.run), "resumed_from": self.resumed_from,
                      "model": None if self.fake else self.client.model,
                      "replies": self.replies, "usd": round(self.spent, 4), "seconds": round(time.time() - started),
                      "walls": self.walls_log, "tool_calls": self.session.calls,
                      "wall_state": self.session.walls}
            (self.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
        return report

    # -------------------------------------------------------------- fake model
    _fake_state: dict = {}

    def _fake_reply(self, items, wall):
        """Scripted replies: declare, make, finish; the wrap-up paints and commits. No network."""
        st = self._fake_state.setdefault(wall, {"step": 0})
        st["step"] += 1
        self.replies += 1
        s = self.session

        def call(name, **args):
            return {"output": [{"type": "function_call", "call_id": f"c{self.replies}", "name": name,
                                "arguments": json.dumps(args), "status": "completed"}],
                    "calls": [{"call_id": f"c{self.replies}", "name": name, "arguments": json.dumps(args)}],
                    "text": "", "usage": {}, "cost": 0.0, "seconds": 0.0, "model": "fake"}
        if wall == "wrapup":
            if st["step"] == 1:
                code = ["import bpy"] + [f"bpy.ops.sg.paint(seg={sid!r}, face={f!r}, rgb=(205,200,190), item='w')"
                                         for sid, faces in s.wrapup_faces().items() for f in faces]
                return call("execute_blender_code", code="\n".join(code), description="paint unseen faces")
            return call("commit")
        seg = s.segments[wall]
        n = seg["number"]
        frames = [f["frame"] for f in seg["frames"]]
        if s.is_floor(wall):         # floors: tiles on the top, no trims
            if st["step"] == 1:
                return call("declare", n=n, findings={"surface": {"rgb": [222, 214, 196], "material": "tiles",
                                                                 "basis": frames}, "trims": []})
            if st["step"] == 2:
                return call("execute_blender_code", description="tiles",
                            code=f"import bpy\nbpy.ops.sg.tiles(seg={wall!r}, face='top', size=(0.3, 0.3), joint=0.004, "
                                 f"rgb=(222,214,196), joint_rgb=(170,160,140), angle=20.0, item='t1')")
            return call("finish_segment", n=n, summary="fake run")
        if st["step"] == 1:          # a wall no photo shows: paint only, empty basis, no trim
            trims = [{"kind": "skirting", "z": [0, 0.1], "rgb": [150, 100, 60], "basis": frames}] if frames else []
            if seg["openings"]:
                trims.append({"kind": "casing", "rgb": [150, 100, 60], "basis": frames})
            return call("declare", n=n, findings={"surface": {"rgb": [200, 192, 178], "material": "paint",
                                                             "basis": frames}, "trims": trims})
        if st["step"] == 2:
            code = [f"import bpy", f"bpy.ops.sg.paint(seg={wall!r}, face='front', rgb=(200,192,178), item='t1')",
                    f"bpy.ops.sg.strip(seg={wall!r}, z0=0.0, z1=0.1, depth=0.015, rgb=(150,100,60), kind='skirting', item='t2')"]
            if not frames:
                code = code[:2]
            if seg["openings"]:
                code += [f"bpy.ops.sg.paint(seg={wall!r}, face='reveal', rgb=(200,192,178), item='t1')",
                         f"bpy.ops.sg.frame(seg={wall!r}, width=0.06, depth=0.015, sides='BOTH', rgb=(150,100,60), item='t3')"]
            return call("execute_blender_code", code="\n".join(code), description="paint and trims")
        return call("finish_segment", n=n, summary="fake run")


class BudgetExceeded(Exception):
    pass


def final_compare(run: Path, out: Path) -> None:
    """Photo | finished model for every key frame of the handoff, into <out>/final_compare/."""
    from . import compare
    plan = json.loads((run / "plan.json").read_text())
    handoff = Path(plan["handoff_dir"])
    doc = json.loads((handoff / "walls.json").read_text())
    frames = sorted({k["source_frame_id"] for w in doc["walls"] for k in w["key_frames"]})
    if (out / "appearance.blend").is_file() and frames:
        compare.main([str(out / "appearance.blend"), *map(str, frames), "--out", str(out / "final_compare"),
                      "--handoff", str(handoff)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?", default=None)
    parser.add_argument("--fake", action="store_true", help="scripted replies, no network, no cost")
    parser.add_argument("--max-usd", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None,
                        help="a replay dir (structuregen.agent.replay --stop-before wrapup): keep its walls, work the rest")
    args = parser.parse_args(argv)
    run = (args.run or (config.OUTPUT_ROOT / config.SCENE / "structuregen/latest")).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or run / f"{'agent_fake' if args.fake else 'agent'}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    Run._last_costs = []
    Run._fake_state = {}
    report = Run(run, out, args.fake, args.max_usd, args.resume.resolve() if args.resume else None).run()
    final_compare(run, out)          # run: the white model; out may be anywhere
    print(json.dumps({k: report[k] for k in ("status", "model", "replies", "usd", "seconds", "tool_calls")}))
    for w in report["walls"]:
        print(f"  {w['wall']:8s} {w['status']:11s} replies {w['replies']}")
    print("AGENT_OUT", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
