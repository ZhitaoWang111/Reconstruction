"""Segment a run of frames, several consecutive frames per conversation.

    $SAM_PYTHON -m src.structure.segment.window --run struct_v4_win5 --frames-every 2

One conversation sees a window of WINDOW consecutive frames, so the model can settle what
a large flat surface is from the neighbouring views before it labels it (a cupboard side
that fills one frame shows its top in the next). Every mask still belongs to one frame.
Windows run in parallel threads; SAM is shared under a lock.

Settings of the run accepted for scene0000_00 (struct_v4_win5): every second frame of the
100-frame sample (50 frames), windows of 5, `coding-openai/gpt-6-astra` at its default
reasoning effort, prompt structure_v4, up to 30 rounds per window. Those are the defaults.

Output: artifacts/structure/<run>/segmentation/ (see the package docstring). A run is
resumable - finished windows are skipped - and refuses to continue if the model, prompt,
frames, vocabulary or window size differ from what it started with.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.harness.client import Client, image as image_part

from ..frames import FRAMES, OUT_ROOT, frame_slots
from src.perception.point_prompt.clicks import (POINT, SAM_CHECKPOINT, Frame, Sam, draw_grid, kept_overview,
                                                tiles_sheet, titled)

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
PROMPT = PROMPTS / "structure_v4.txt"
MODEL = "coding-openai/gpt-6-astra"
VOCAB = ("wall", "floor", "door", "window")
WINDOW = 5
MAX_ROUNDS = 30
SEGMENT_MAX = 8
OVERLAP_NOTE = ("if {outer} is a wall or floor mask it has spilled over {inner}: segment {outer} again "
                "with negative points there, or cut {inner} out of it with combine")
FRAME_ID = {"type": "integer", "description": "frame id, one of the ids in this window"}


def fn(name, description, properties, required=None):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required or list(properties),
        "additionalProperties": False}}}


TOOLS = [
    fn("plan", "Declare what you will segment, each in one frame. Returns plan ids.",
       {"objects": {"type": "array", "minItems": 1, "items": {
           "type": "object", "additionalProperties": False, "required": ["frame", "label", "where"],
           "properties": {"frame": FRAME_ID, "label": {"type": "string"},
                          "where": {"type": "string", "description": "where it is in that frame, a few words"}}}}}),
    fn("segment", f"Run SAM on up to {SEGMENT_MAX} planned items, from any frames of the window. Each item: plan_id, "
                  "positive points, optional negative points, as [x, y] in pixels of that item's frame. "
                  "Returns one candidate per item, drawn as tiles.",
       {"items": {"type": "array", "minItems": 1, "maxItems": SEGMENT_MAX, "items": {
           "type": "object", "additionalProperties": False, "required": ["plan_id", "positive", "negative"],
           "properties": {"plan_id": {"type": "string"},
                          "positive": {"type": "array", "minItems": 1, "maxItems": 20, "items": POINT},
                          "negative": {"type": "array", "maxItems": 8, "items": POINT}}}}}),
    fn("zoom", "Magnified crop of one frame with a finer grid labelled in that frame's pixels.",
       {"frame": FRAME_ID, "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4,
                                   "description": "[x0, y0, x1, y1] in pixels of that frame"}}),
    fn("combine", "Make one candidate for a plan id from the union of its candidates, minus candidates to subtract "
                  "(any candidates of the same frame).",
       {"plan_id": {"type": "string"}, "union": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "subtract": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string", "description": "what this fixes, a few words"}}),
    fn("keep", "Accept candidates (C ids). A later keep for the same plan id replaces the earlier one.",
       {"candidates": {"type": "array", "minItems": 1, "items": {"type": "string"}}}),
    fn("finish", "End the window. Every plan id must be kept or listed here with a reason. Also list here, with "
                 "plan_id \"-\", any large surface you decided not to plan and why.",
       {"skipped": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                "required": ["plan_id", "reason"],
                                                "properties": {"plan_id": {"type": "string"},
                                                               "reason": {"type": "string"}}}}}),
]


class Window:
    """One conversation over several frames: routes each tool call to the frame it names."""

    def __init__(self, index, fids, sam, run_dir: Path, frames_dir: Path = FRAMES, vocab=VOCAB, model=MODEL):
        self.index, self.fids, self.sam = index, list(fids), sam
        self.out = run_dir / f"window_{index:02d}_{fids[0]:06d}-{fids[-1]:06d}"
        (self.out / "images").mkdir(parents=True, exist_ok=True)
        self.ids = {"P": 0, "C": 0}
        self.frames = {fid: Frame(fid, sam, run_dir / f"frame_{fid:06d}", frames_dir, vocab, self.ids, model)
                       for fid in fids}
        self.plan_frame, self.cand_frame = {}, {}
        self.not_planned = []
        self.finished = False
        self.n_img = 0

    def save_img(self, img, name):
        self.n_img += 1
        p = self.out / "images" / f"{self.n_img:02d}_{name}.jpg"
        img.save(p, quality=90)
        return p

    def frame(self, fid) -> Frame:
        if fid not in self.frames:
            raise ValueError(f"{fid} is not a frame of this window; the frames are {self.fids}")
        return self.frames[fid]

    def of_plan(self, pid) -> Frame:
        if pid not in self.plan_frame:
            raise ValueError(f"{pid} is not a plan id; call plan first")
        return self.frames[self.plan_frame[pid]]

    def of_cand(self, cid) -> Frame:
        if cid not in self.cand_frame:
            raise ValueError(f"{cid} is not a candidate id")
        return self.frames[self.cand_frame[cid]]

    def first_message(self):
        fr0 = next(iter(self.frames.values()))
        content = [{"type": "text", "text": (
            f"A window of {len(self.fids)} consecutive frames from one walk through a room, in capture order: "
            f"frame ids {self.fids}. Each is {fr0.W}x{fr0.H} pixels and is shown with a grid every 100 pixels "
            f"(labels are x along the top, y down the left).")}]
        for fid, fr in self.frames.items():
            p = fr.save_img(draw_grid(fr.img, 0, 0, fr.W, fr.H, 100, 1.0), "grid")
            content += [{"type": "text", "text": f"Frame {fid}:"}, image_part(p)]
        return content

    # tools -------------------------------------------------------------
    def t_plan(self, objects):
        for o in objects:                           # validate everything before any id is handed out
            fr = self.frame(int(o["frame"]))
            if o["label"].strip() not in fr.vocab:
                raise ValueError(f"label {o['label']!r} is not allowed; the label must be exactly one of {fr.vocab}")
        ids = []
        for o in objects:
            fid = int(o["frame"])
            pid = self.frames[fid].plan_item(o["label"], o["where"])
            self.plan_frame[pid] = fid
            ids.append({"plan_id": pid, "frame": fid, "label": o["label"].strip(), "where": o["where"].strip()})
        return {"planned": ids, "total_planned": len(self.plan_frame)}, []

    def t_segment(self, items):
        groups = {}
        for it in items:
            groups.setdefault(self.of_plan(it["plan_id"]).fid, []).append(it)
        rows, tiles = [], []
        for fid, group in groups.items():
            fr = self.frames[fid]
            with self.sam.lock:
                self.sam.set_image(fr.img, fid)
                r, t = fr.segment_rows(group)
            for row in r:
                self.cand_frame[row["candidate"]] = fid
                rows.append({"frame": fid, **row})
            tiles += t
        p = self.save_img(tiles_sheet(tiles, cols=4), "segment")
        return {"candidates": rows,
                "image": "One tile per candidate, in order, titled with its frame: mask tinted blue with cyan outline, "
                         "rest dimmed; green dots = positive points, red crosses = negative points."}, [p]

    def t_zoom(self, frame, box):
        res, imgs = self.frame(int(frame)).zoom(box)
        return {"frame": int(frame), **res}, imgs

    def t_combine(self, plan_id, union, subtract=None, reason=""):
        fr = self.of_plan(plan_id)
        for cid in list(union) + list(subtract or []):
            if self.of_cand(cid).fid != fr.fid:
                raise ValueError(f"{cid} belongs to frame {self.cand_frame[cid]}, but {plan_id} is in frame {fr.fid}")
        res, imgs = fr.combine(plan_id, list(union), list(subtract or []), reason or "combined")
        self.cand_frame[res["candidate"]] = fr.fid
        return {"frame": fr.fid, **res}, imgs

    def t_keep(self, candidates):
        groups = {}
        for cid in candidates:
            groups.setdefault(self.of_cand(cid).fid, []).append(cid)
        done, warnings = [], []
        for fid, group in groups.items():
            kept, overlaps = self.frames[fid].keep(group, OVERLAP_NOTE)
            done += [f"{cid} -> {self.frames[fid].cands[cid]['plan_id']} (frame {fid})" for cid in kept]
            warnings += [f"frame {fid}: {w}" for w in overlaps]
        sheet = tiles_sheet([titled(kept_overview(self.frames[fid].rgb, self.frames[fid].kept), f"frame {fid}: kept so far")
                             for fid in groups], cols=3)
        p = self.save_img(sheet, "kept")
        todo = [pid for pid, fid in self.plan_frame.items() if pid not in self.frames[fid].kept]
        return {"kept": done, "kept_total": sum(len(f.kept) for f in self.frames.values()),
                "planned_not_kept": todo, "overlap_warnings": warnings,
                "image": "Every kept mask so far in the frames you just touched, labelled by plan id."}, [p]

    def t_finish(self, skipped):
        notes = [s["reason"] for s in skipped if s["plan_id"].strip() in ("-", "")]
        real = [s for s in skipped if s["plan_id"].strip() not in ("-", "")]
        by_frame = {fid: [] for fid in self.fids}
        for s in real:
            by_frame[self.of_plan(s["plan_id"]).fid].append(s)
        problems = []
        for fid, fr in self.frames.items():
            sk = {s["plan_id"] for s in by_frame[fid]}
            both = [pid for pid in sk if pid in fr.kept]
            missing = fr.unresolved(sk)
            if both:
                problems.append(f"frame {fid}: {both} are kept; do not also skip them")
            if missing:
                problems.append(f"frame {fid}: neither kept nor skipped: {missing}")
        if problems:
            raise ValueError("; ".join(problems))
        for fid, fr in self.frames.items():
            fr.finish(by_frame[fid])
        self.not_planned = notes
        self.finished = True
        return {"finished": True, "kept": sum(len(f.kept) for f in self.frames.values()), "skipped": len(real)}, []

    def call(self, name, args):
        if not any(t["function"]["name"] == name for t in TOOLS):
            raise ValueError(f"unknown tool {name!r}")
        return getattr(self, "t_" + name)(**args)


def run_window(index, fids, sam, run_dir, prompt, max_rounds, model, effort, frames_dir=FRAMES, vocab=VOCAB):
    win = Window(index, fids, sam, run_dir, frames_dir, vocab, model)
    client = Client(model, win.out, timeout=300, reasoning_effort=effort)
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": win.first_message()}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.monotonic()
    rounds = 0
    for step in range(max_rounds):
        rounds = step + 1
        if step == max_rounds - 2:
            messages.append({"role": "user", "content": "Two rounds left: keep what is good now, then call finish, "
                                                        "skipping what is not kept with a reason."})
        resp = client.call(messages, TOOLS, f"r{step:02d}")
        meta = sorted((win.out / "calls").glob(f"r{step:02d}_*_meta.json"))
        if meta:
            u = json.loads(meta[-1].read_text()).get("usage") or {}
            for k in usage:
                usage[k] += u.get(k) or 0
        messages.append(resp)
        call = resp["tool_calls"][0]
        name = call["function"]["name"]
        try:
            result, images = win.call(name, json.loads(call["function"]["arguments"]))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            result, images = {"error": str(exc)}, []
        print(f"  [w{index:02d}] r{step:02d} {name}: {json.dumps(result, ensure_ascii=False)[:150]}", flush=True)
        (win.out / "calls" / f"r{step:02d}_tool.json").write_text(json.dumps(
            {"name": name, "arguments": call["function"]["arguments"], "result": result,
             "images": [str(p) for p in images]}, ensure_ascii=False, indent=2))
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
        for extra in resp["tool_calls"][1:]:
            messages.append({"role": "tool", "tool_call_id": extra["id"],
                             "content": json.dumps({"error": "one tool call per reply; this one was ignored"})})
        if images:
            messages.append({"role": "user", "content": [{"type": "text", "text": f"Image(s) from {name}:"}]
                             + [image_part(p) for p in images]})
        if win.finished:
            break
    seconds = time.monotonic() - t0
    n = len(fids)
    share = {k: v // n for k, v in usage.items()}                # each frame's equal share of its window
    extra = {"window": index, "window_frames": list(fids), "window_rounds": rounds, "window_usage": usage,
             "window_seconds": round(seconds, 1), "usage_is_window_share": True, "not_planned_notes": win.not_planned}
    for fr in win.frames.values():
        fr.finished = win.finished
        fr.write(rounds, share, seconds / n, extra)
    result = {"window": index, "frames": list(fids), "finished": win.finished, "rounds": rounds, "usage": usage,
              "seconds": round(seconds, 1), "kept": sum(len(f.kept) for f in win.frames.values()),
              "skipped": sum(len(f.skipped) for f in win.frames.values()), "not_planned_notes": win.not_planned,
              "model": model}
    (win.out / "window.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def one_window(index, fids, sam, run_dir, prompt, max_rounds, model, effort):
    folder = run_dir / f"window_{index:02d}_{fids[0]:06d}-{fids[-1]:06d}"
    try:
        r = run_window(index, fids, sam, run_dir, prompt, max_rounds, model, effort)
        (folder / "error.json").unlink(missing_ok=True)
        print(f"DONE w{index:02d} {fids}: finished={r['finished']} rounds={r['rounds']} kept={r['kept']} "
              f"tokens={r['usage']} {r['seconds']}s", flush=True)
        return r
    except Exception as exc:  # noqa: BLE001 - one bad window must not stop the batch
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "error.json").write_text(json.dumps({"error": repr(exc), "trace": traceback.format_exc()}, indent=2))
        print(f"FAILED w{index:02d} {fids}: {exc!r}", flush=True)
        return None


def segmentation_dir(run: str) -> Path:
    return OUT_ROOT / run / "segmentation"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--frames", type=int, nargs="+", help="frame ids in capture order")
    pick.add_argument("--frames-every", type=int, default=2,
                      help="every n-th frame of the AMB3R sample (default 2: 50 of 100)")
    ap.add_argument("--window", type=int, default=WINDOW, help="frames per conversation")
    ap.add_argument("--prompt", type=Path, default=PROMPT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help="per window")
    ap.add_argument("--workers", type=int, default=4, help="windows in flight; SAM is shared under a lock")
    ap.add_argument("--sam-checkpoint", type=Path, default=SAM_CHECKPOINT)
    args = ap.parse_args()
    frames = args.frames or sorted(frame_slots())[::args.frames_every]
    run_dir = segmentation_dir(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text()
    windows = [frames[i:i + args.window] for i in range(0, len(frames), args.window)]
    config = {"model": args.model, "reasoning_effort": args.reasoning_effort, "sam_checkpoint": str(args.sam_checkpoint),
              "max_rounds": args.max_rounds, "window": args.window, "prompt_file": args.prompt.name, "vocab": list(VOCAB),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "frames": frames}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        for key in ("model", "prompt_sha256", "frames", "max_rounds", "vocab", "window"):
            if old.get(key) != config[key]:
                raise SystemExit(f"{key} differs from the existing run in {run_dir}; choose a new --run")
    cfg_path.write_text(json.dumps({**config, "workers": args.workers}, indent=2))
    (run_dir / "prompt.txt").write_text(prompt)
    todo = [(i, w) for i, w in enumerate(windows)
            if not (run_dir / f"window_{i:02d}_{w[0]:06d}-{w[-1]:06d}" / "window.json").exists()]
    print(f"{len(frames)} frames in {len(windows)} windows of {args.window}; {len(todo)} to run, "
          f"{args.workers} workers, model {args.model}", flush=True)
    sam = Sam(args.sam_checkpoint)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda iw: one_window(iw[0], iw[1], sam, run_dir, prompt, args.max_rounds,
                                            args.model, args.reasoning_effort), todo))
    summary = {"frames": len(frames), "windows": len(windows), "finished": 0, "unfinished": 0, "error": [],
               "kept": 0, "skipped": 0, "rounds": 0, "prompt_tokens": 0, "completion_tokens": 0,
               "this_invocation_seconds": round(time.monotonic() - t0, 1)}
    for i, w in enumerate(windows):
        f = run_dir / f"window_{i:02d}_{w[0]:06d}-{w[-1]:06d}" / "window.json"
        if not f.exists():
            summary["error"].append(i)
            continue
        r = json.loads(f.read_text())
        summary["finished" if r["finished"] else "unfinished"] += 1
        summary["kept"] += r["kept"]; summary["skipped"] += r["skipped"]; summary["rounds"] += r["rounds"]
        summary["prompt_tokens"] += r["usage"]["prompt_tokens"]
        summary["completion_tokens"] += r["usage"]["completion_tokens"]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
