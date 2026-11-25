"""The appearance session: the agent's tools, the per-wall state and the gates behind them.

One `Session` drives one Blender worker (`blender_worker.py`, headless) for a whole run. The agent
loop (or a scripted run) calls the tool methods; each returns a dict with `text` for the agent and,
where there is one, `image` (a compare picture path). Every call is appended to `transcript.jsonl`.

Tools, in the order a wall uses them:
    declare(n, findings)                  what the wall looks like and on which photos; makes items
    execute_blender_code(code, desc)      the only edit; then automatic checks and a compare picture
    compare(frame)                        photo | current model from that photo's camera
    finish_segment(n, summary, dropped)   final check of this wall; on success the next wall
Gates: no edit before declare; agent code passes mcp-for-blender's safe-mode validator; a code
error or a hard problem rolls the edit back (in the worker); finish needs no hard or trim
problem, every room-side face of the wall painted, and every item done, dropped with a reason,
or retired after MAX_ITEM_TRIES edits.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2

from .. import config
from ..common import handoff as handoff_io
from . import compare as cmp

MAX_ITEM_TRIES = 3              # an item edited this many times and still open stops blocking
TRIM_KINDS = ("skirting", "rail", "casing")
WORKER = Path(__file__).parent / "blender" / "blender_worker.py"
SAFE_MODE = config.BLENDER_MCP / "src/blender_mcp/safe_mode.py"


def _safe_mode():
    spec = importlib.util.spec_from_file_location("sg_mcp_safe_mode", SAFE_MODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ToolError(Exception):
    """A refused call: the message goes back to the agent, nothing changed."""


class Session:
    def __init__(self, run: Path, out: Path, order: list[str] | None = None,
                 start: Path | None = None, done: dict | None = None):
        """``start``/``done``: resume from a scene whose bodies in ``done`` (their wall state, e.g. a replay's
        walls_done.json) are already finished; only the others are worked, then the wrap-up. The white-model
        geometry check still holds: painting and trims never change a body's mesh."""
        self.run, self.out = Path(run), Path(out)
        if self.out.exists():
            shutil.rmtree(self.out)
        (self.out / "compare").mkdir(parents=True)
        self.card = json.loads((self.run / "evidence/task_card.json").read_text())
        plan = json.loads((self.run / "plan.json").read_text())
        self.handoff = Path(plan["handoff_dir"])
        doc = handoff_io.load(self.handoff)
        self.photo = {k["source_frame_id"]: self.handoff / k["rgb"] for w in doc["walls"] for k in w["key_frames"]}
        self.segments = {s["id"]: s for s in self.card["segments"]}
        self.by_number = {s["number"]: s["id"] for s in self.card["segments"]}
        # floors are worked like segments, first: one room = one floor finish, simpler than a wall,
        # and once laid the walls' compare pictures show a real floor under the skirting
        self.floors = {f["id"]: {**f, "number": f["id"], "role": "floor", "openings": []}
                       for f in self.card.get("floors", [])}
        self.segments.update(self.floors)
        first = self.card["first"]["segment"]
        # walls no photo shows go last, so the walls they take their paint from are finished first
        rest = [s for s in self.card["segments"] if s["id"] != first]
        floor_order = [f for f, x in self.floors.items() if x["kind"] == "room" and x["frames"]] + \
                      [f for f, x in self.floors.items() if x["kind"] == "room" and not x["frames"]] + \
                      [f for f, x in self.floors.items() if x["kind"] != "room"]
        self.order = order or floor_order + [first] + [s["id"] for s in rest if s["frames"]] + \
            [s["id"] for s in rest if not s["frames"]]
        self.safe = _safe_mode()
        self.walls: dict[str, dict] = dict(done or {})
        self.order = [x for x in self.order if x not in self.walls]
        self.current: str | None = None
        self.calls = 0
        blend = self.out / "session.blend"
        shutil.copyfile(start or self.run / "walls.blend", blend)
        self.proc = subprocess.Popen([str(config.BLENDER), "-b", str(blend), "--python", str(WORKER), "--",
                                      str(self.out / "worker")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.log = open(self.out / "transcript.jsonl", "w")
        if self.order:
            self.start_segment(self.order[0])
        else:
            self.current = "wrapup"

    # ----------------------------------------------------------------- plumbing
    def _worker(self, **cmd) -> dict:
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            if line.startswith("SG_REPLY "):
                reply = json.loads(line[len("SG_REPLY "):])
                if "traceback" in reply:
                    raise RuntimeError(reply["error"] + "\n" + reply["traceback"])
                return reply
        raise RuntimeError("Blender worker exited")

    def _record(self, tool, args, result, segment=None):
        self.calls += 1
        self.log.write(json.dumps({"n": self.calls, "t": round(time.time(), 2), "segment": segment or self.current,
                                   "tool": tool, "args": args, "result": result}, ensure_ascii=False) + "\n")
        self.log.flush()
        return result

    def close(self):
        if self.proc.poll() is None:
            self._worker(cmd="quit")
            self.proc.wait(timeout=30)
        self.log.close()

    # ----------------------------------------------------------------- per wall
    def label(self, sid):
        return f"floor {sid}" if self.is_floor(sid) else f"wall #{self.segments[sid]['number']} {sid}"

    def is_floor(self, sid):
        return sid in getattr(self, "floors", {})

    def faces_required(self, sid):
        if self.is_floor(sid):
            return ["top"]
        return ["front"] + (["reveal"] if self.segments[sid]["openings"] else [])

    ALL_FACES = ("front", "back", "top", "bottom", "end_start", "end_end", "reveal")
    FLOOR_FACES = ("top", "side", "bottom")

    def wrapup_faces(self):
        """Every face of every segment and floor that the per-body passes did not have to paint."""
        out = {}
        for sid, seg in self.segments.items():
            if self.is_floor(sid):
                faces = list(self.FLOOR_FACES)
            else:
                faces = [f for f in self.ALL_FACES if f != "front" and (f != "reveal" or seg["openings"])]
            out[sid] = [f for f in faces if f not in self.faces_required(sid)]
        return out

    def start_segment(self, sid):
        seg = self.segments[sid]
        self.current = sid
        self.walls[sid] = {"status": "open", "declared": None, "items": {}, "edits": 0,
                           "frames": [f["frame"] for f in seg["frames"]]}
        return sid

    def _wall(self, n):
        if self.current not in self.segments:
            raise ToolError(f"no wall is open now ({self.current}); use execute_blender_code / commit")
        sid = self.by_number.get(n) if isinstance(n, int) else (self.by_number.get(int(n)) if str(n).isdigit() else n)
        if sid != self.current:
            raise ToolError(f"now working on {self.label(self.current)}, not {n}")
        return self.walls[sid]

    def opening_brief(self) -> dict:
        """What the harness gives at the start of a wall: its facts, its photos, a compare picture."""
        sid = self.current
        if self.is_floor(sid):
            return self.floor_brief(sid)
        seg = self.segments[sid]
        frames = self.walls[sid]["frames"]
        images = []
        for f in seg["frames"]:
            images += [self.handoff / f["rgb"], self.handoff / f["boxed"]]
        best = self._best_frame(sid)
        pic = self._compare(best) if best is not None else None
        text = (f"Wall #{seg['number']} {sid} ({seg['role']}), length {seg['length_m']} m, "
                f"front normal {seg['normal_room']}, openings {seg['openings'] or 'none'}. "
                f"Key frames: {', '.join(map(str, frames)) or 'none'}."
                + (f" Attached: original and boxed photo for each, and compare({best}) of the current model."
                   if frames else ""))
        if not frames:
            text += " " + self.neighbour_note(sid)
        return self._record("brief", {"segment": sid}, {"text": text, "images": [str(p) for p in images],
                                                         "image": pic})

    def floor_brief(self, sid) -> dict:
        """A floor: its size, the frames that show it (original + outlined), a compare picture."""
        f = self.floors[sid]
        frames = self.walls[sid]["frames"]
        images = []
        for x in f["frames"]:
            images += [self.handoff / x["rgb"], self.run / "evidence" / x["boxed"]]
        best = frames[0] if frames else None
        pic = self._compare(best) if best is not None else None
        if f["kind"] == "room":
            text = (f"Floor {sid} (a room), {f['area_m2']} m2. Key frames: {', '.join(map(str, frames)) or 'none'}."
                    + (f" Attached: original and outlined photo for each (yellow: this floor's edge where no wall hides it),"
                       f" and compare({best}) of the current model." if frames else ""))
            if not frames:
                text += " No key frame shows this floor: take the finish of the room next to it through a doorway."
        else:
            a, b = (f.get("between") or [None, None])[:2]
            text = (f"Floor {sid}: the threshold under the doorway of {f['opening_of']}, {f['area_m2']} m2, between "
                    f"{a} and {b} (both finished). Usually it continues one of the two floors, or is a separate "
                    f"strip (stone, wood). Key frames: {', '.join(map(str, frames)) or 'none'}.")
        return self._record("brief", {"segment": sid}, {"text": text, "images": [str(p) for p in images], "image": pic})

    def neighbours(self, sid) -> dict:
        """Faces that bound the same space as this segment's front, from the plan's geometry alone.

        same_space: segments of another wall whose front meets this front at a shared corner
        (a mitred corner: the two faces see the same space), and the face of a wall that butts
        onto this segment's end and looks along it. across_partition: the segment of the same
        wall on the other side of a butt cut: a partition stands between them.
        """
        plan = json.loads((self.run / "plan.json").read_text())
        segs = {s["id"]: s for s in plan["segments"]}
        me = segs[sid]
        f0, f1 = me["front"]
        close = lambda p, q: abs(p[0] - q[0]) + abs(p[1] - q[1]) < 1e-3
        same, across = [], []
        for other in plan["segments"]:
            if other["id"] == sid:
                continue
            shares = any(close(p, q) for p in other["front"] for q in me["front"])
            if not shares:
                continue
            (across if other["wall"] == me["wall"] else same).append(f"{other['id']} front")
        ux, uy = f1[0] - f0[0], f1[1] - f0[1]
        n = (ux * ux + uy * uy) ** 0.5
        u = (ux / n, uy / n)
        for other in plan["segments"]:
            if other["wall"] == me["wall"] or other["role"] != "interior" or \
                    any(close(p, q) for p in other["front"] for q in me["front"]):       # mitred neighbour, not a butt
                continue
            for end, inward in ((f0, u), (f1, (-u[0], -u[1]))):
                # a wall butting here has one end on this front face, at this end of the segment
                if not any(abs((p[0] - end[0]) * u[1] - (p[1] - end[1]) * u[0]) < 0.02 and
                           abs((p[0] - end[0]) * u[0] + (p[1] - end[1]) * u[1]) < other["thickness_m"] + 0.02
                           for p in other["front"] + other["back"]):
                    continue
                o0, o1 = other["front"]
                ox, oy = o1[0] - o0[0], o1[1] - o0[1]
                room = (-oy, ox) if ((other["back"][0][0] - o0[0]) * -oy + (other["back"][0][1] - o0[1]) * ox) < 0 else (oy, -ox)
                face = "front" if room[0] * inward[0] + room[1] * inward[1] > 0 else "back"
                same.append(f"{other['id']} {face}")
        return {"same_space": sorted(set(same)), "across_partition": sorted(set(across))}

    def neighbour_note(self, sid) -> str:
        nb = self.neighbours(sid)
        note = ("No key frame shows this wall, so its colour must come from the faces that bound the same space: "
                + (", ".join(nb["same_space"]) or "none found") + ".")
        if nb["across_partition"]:
            note += (" Not from " + ", ".join(nb["across_partition"]) + ": that is the same wall on the other side "
                     "of a partition, a different room.")
        return note

    def _best_frame(self, sid):
        seg = self.segments[sid]
        if not seg["frames"]:
            return None
        if self.is_floor(sid):
            return seg["frames"][0]["frame"]
        return min(seg["frames"], key=lambda f: (f["view_angle_deg"], f["reprojection_px"] or 0))["frame"]

    # ----------------------------------------------------------------- tools
    def declare(self, n, findings: dict) -> dict:
        args = {"n": n, "findings": findings}
        try:
            wall = self._wall(n)
            frames = set(wall["frames"])
            surface = findings.get("surface") or {}
            trims = findings.get("trims") or []
            problems = []
            if not surface.get("basis") and frames:
                problems.append("surface: give the frames it is based on (basis)")
            bad = [f for f in surface.get("basis", []) if f not in frames]
            if bad:
                problems.append(f"surface: frames {bad} are not key frames of this wall {sorted(frames)}")
            if self.is_floor(self.current) and trims:
                problems.append("a floor has one finish and no trims: leave trims empty")
            for i, t in enumerate(trims):
                if t.get("kind") not in TRIM_KINDS:
                    problems.append(f"trims[{i}]: kind must be one of {TRIM_KINDS} (door leaves, windows and "
                                    f"window casings are made by object generation, not here)")
                if (frames and not t.get("basis")) or any(f not in frames for f in t.get("basis", [])):
                    problems.append(f"trims[{i}]: basis must list key frames of this wall {sorted(frames)}")
                if t.get("kind") == "casing" and not self.segments[self.current]["openings"]:
                    problems.append(f"trims[{i}]: this wall has no handoff opening, so no casing")
            if problems:
                raise ToolError("declare refused: " + "; ".join(problems))
            old = wall["items"]
            items = {"t1": {"what": "surface", "spec": surface}}
            for i, t in enumerate(trims, start=2):
                items[f"t{i}"] = {"what": t["kind"], "spec": t}
            for k, v in items.items():
                prev = old.get(k)
                same = prev is not None and prev["spec"] == v["spec"]
                v.update(tries=prev["tries"] if same else 0, status=prev["status"] if same else "open",
                         reason=prev.get("reason") if same else None)
            wall["items"], wall["declared"] = items, findings
            listing = "; ".join(f"{k} {v['what']}" for k, v in items.items())
            return self._record("declare", args, {"text": f"Recorded. Items for this wall: {listing}."})
        except ToolError as e:
            return self._record("declare", args, {"text": str(e), "refused": True})

    def execute_blender_code(self, code: str, description: str) -> dict:
        args = {"code": code, "description": description}
        sid = self.current
        if sid == "wrapup":
            return self._wrapup_edit(code, description, args)
        if sid not in self.walls:
            return self._record("execute_blender_code", args, {"text": "refused: nothing to edit now", "refused": True})
        wall = self.walls[sid]
        if wall["declared"] is None:
            return self._record("execute_blender_code", args, {"text": "refused: call declare for this wall first",
                                                                "refused": True})
        try:
            self.safe.validate_code(code)
        except self.safe.SandboxViolation as e:
            return self._record("execute_blender_code", args, {"text": f"refused by safe mode: {e}", "refused": True})
        r = self._worker(cmd="edit", code=code, segment=sid)
        touched = set(re.findall(r"item\s*=\s*['\"](t\d+)['\"]", code))
        if r["ok"]:
            wall["edits"] += 1
            for k in touched & set(wall["items"]):
                wall["items"][k]["tries"] += 1
        progress = self._progress(sid)
        lines = []
        if r["rolled_back"]:
            why = r["error"] or "; ".join(r["hard"])
            lines.append(f"ROLLED BACK, the scene is as before this call: {why}")
        else:
            lines.append(f"Done ({description}).")
        if r["trims"]:
            lines.append("Trim problems (fix before finishing): " + "; ".join(r["trims"]))
        lines.append(self._progress_line(progress))
        if r["stdout"].strip():
            lines.append("stdout: " + r["stdout"].strip()[-600:])
        best = self._best_frame(sid)
        pic = self._compare(best) if best is not None and not r["rolled_back"] else None
        return self._record("execute_blender_code", args, {"text": "\n".join(lines), "image": pic,
                                                            "rolled_back": r["rolled_back"], "trims": r["trims"]})

    def _wrapup_edit(self, code, description, args):
        try:
            self.safe.validate_code(code)
        except self.safe.SandboxViolation as e:
            return self._record("execute_blender_code", args, {"text": f"refused by safe mode: {e}", "refused": True})
        r = self._worker(cmd="edit", code=code, segment=list(self.segments),
                         keep=[[sid, f"{sid}_{f}"] for sid in self.segments for f in self.faces_required(sid)])
        lines = [f"ROLLED BACK, the scene is as before this call: {r['error'] or '; '.join(r['hard'])}"
                 if r["rolled_back"] else f"Done ({description})."]
        if r["trims"]:
            lines.append("Trim problems: " + "; ".join(r["trims"]))
        left = []
        for sid, faces in self.wrapup_faces().items():
            p = self._worker(cmd="progress", segment=sid, items=[], faces=faces)
            if p["unpainted"]:
                left.append(f"{sid}: {', '.join(p['unpainted'])}")
        lines.append("Still without colour: " + ("; ".join(left) if left else "none, call commit()"))
        return self._record("execute_blender_code", args, {"text": "\n".join(lines), "rolled_back": r["rolled_back"]})

    def compare(self, frame: int) -> dict:
        if self.current not in self.walls:
            frames = sorted(self.photo)
            if frame not in frames:
                return self._record("compare", {"frame": frame}, {"text": f"refused: key frames are {frames}", "refused": True})
            return self._record("compare", {"frame": frame}, {"text": f"compare({frame})", "image": self._compare(frame)})
        if frame not in self.walls[self.current]["frames"]:
            return self._record("compare", {"frame": frame}, {
                "text": f"refused: {frame} is not a key frame of this wall {self.walls[self.current]['frames']}",
                "refused": True})
        return self._record("compare", {"frame": frame}, {"text": f"compare({frame})", "image": self._compare(frame)})

    def finish_segment(self, n, summary: str, dropped: dict | None = None) -> dict:
        args = {"n": n, "summary": summary, "dropped": dropped or {}}
        try:
            wall = self._wall(n)
        except ToolError as e:
            return self._record("finish_segment", args, {"text": str(e), "refused": True})
        for k, reason in (dropped or {}).items():
            if k in wall["items"] and reason and len(reason.strip()) >= 4:
                wall["items"][k]["status"], wall["items"][k]["reason"] = "dropped", reason.strip()
        check = self._worker(cmd="check", segment=self.current)
        progress = self._progress(self.current)
        blockers = list(check["hard"]) + list(check["trims"])
        if progress["unpainted"]:
            blockers.append("faces still without a colour: " + ", ".join(progress["unpainted"]))
        open_items = [k for k in progress["items_open"] if wall["items"][k]["status"] == "open"]
        if open_items:
            blockers.append("items not done (do them, or drop them with a reason): " + ", ".join(open_items))
        if blockers:
            return self._record("finish_segment", args, {"text": "Not finished:\n- " + "\n- ".join(blockers),
                                                          "refused": True})
        wall.update(status="done", summary=summary)
        done = self.current
        nxt = next((s for s in self.order if s not in self.walls), None)
        text = f"{self.label(self.current)} finished."
        if nxt:
            self.start_segment(nxt)
            text += f" Next: {self.label(nxt)}."
        else:
            self.current = "wrapup"
            text += (" All walls done. Wrap-up: paint the faces no photo shows (back faces, tops, ends, "
                     "the bathroom side of the inner walls), then call commit().")
        return self._record("finish_segment", args, {"text": text, "next": nxt}, segment=done)

    def wrapup_brief(self) -> dict:
        faces = self.wrapup_faces()
        done = {sid: w.get("summary") for sid, w in self.walls.items() if w.get("status") in ("done", "unconverged")}
        listing = "; ".join(f"{sid}: {', '.join(f)}" for sid, f in faces.items())
        text = ("Wrap-up. No photo shows these faces, so their colour can only be inferred from the walls and floors "
                "done or assumed. Paint each one with bpy.ops.sg.paint(seg=..., face=..., rgb=..., roughness=..., item='w')."
                " Any segment or floor may be edited now; walls' room-side faces, floors' tops and trims must not change.\n"
                f"Faces to paint: {listing}.\nWall summaries: " + json.dumps(done, ensure_ascii=False))
        return self._record("brief", {"segment": "wrapup"}, {"text": text})

    def commit(self) -> dict:
        """Whole-scene final check. In the wrap-up only; returns ok or the list of what blocks it."""
        if self.current != "wrapup":
            return self._record("commit", {}, {"text": "refused: finish every wall first", "refused": True})
        blockers = []
        for sid, wall in self.walls.items():
            if wall["status"] not in ("done", "unconverged"):
                blockers.append(f"{sid} not finished")
        check = self._worker(cmd="check", segment=list(self.segments))
        blockers += check["hard"] + check["trims"]
        for sid, faces in self.wrapup_faces().items():
            p = self._worker(cmd="progress", segment=sid, items=[], faces=faces + self.faces_required(sid))
            if p["unpainted"]:
                blockers.append(f"{sid}: no colour yet on {', '.join(p['unpainted'])}")
        if blockers:
            return self._record("commit", {}, {"text": "Not committed:\n- " + "\n- ".join(blockers), "refused": True})
        self.current = "committed"
        return self._record("commit", {}, {"text": "Committed: every face has a colour, no problems."})

    # ----------------------------------------------------------------- helpers
    def _progress(self, sid):
        wall = self.walls[sid]
        p = self._worker(cmd="progress", segment=sid, items=list(wall["items"]), faces=self.faces_required(sid))
        for k in p["items_done"]:
            if wall["items"][k]["status"] == "open":
                wall["items"][k]["status"] = "done"
        for k in p["items_open"]:
            item = wall["items"][k]
            if item["status"] == "open" and item["tries"] >= MAX_ITEM_TRIES:
                item["status"], item["reason"] = "unconverged", f"still open after {item['tries']} edits"
        p["items_open"] = [k for k in p["items_open"] if wall["items"][k]["status"] == "open"]
        return p

    def _progress_line(self, p):
        wall = self.walls[self.current]
        parts = [f"{k} {v['what']}: {v['status']}" for k, v in wall["items"].items()]
        line = "Items: " + "; ".join(parts) + "."
        if p["unpainted"]:
            line += " Faces without colour: " + ", ".join(p["unpainted"]) + "."
        return line

    def _compare(self, frame) -> str:
        tmp = self.out / "worker/passes"
        self._worker(cmd="passes", cams=[f"frame_{frame:04d}"], out=str(tmp), scale=cmp.SCALE)
        img = cmp.compose(cv2.imread(str(self.photo[frame])), cmp.model_image(tmp / f"frame_{frame:04d}"),
                          f"frame {frame}   left: photo   right: current model from the same camera")
        path = self.out / "compare" / f"{self.calls + 1:03d}_{frame:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return str(path)

    def save(self, path: Path):
        self._worker(cmd="save", path=str(path))
