"""Point-prompt object segmentation, one frame per conversation.

    $SAM31_PYTHON -m src.perception.point_prompt.objects --run full100_v2

The model sees the frame twice (as captured, and with a 100 px grid), plans the objects in
it with free labels, clicks points for each, and keeps, retries, combines or skips. One
conversation per frame; frames run in parallel threads with SAM shared under a lock.

Defaults are the run that fed harness-v15 (full100_v2): the 100 frames of
scene_amb3r_full100_v1, `openai/gpt-6-sol`, prompt point_objects_sep_v5 (boundaries follow
structural breaks), up to 24 rounds per frame, 4 frames in flight.

Output per run: artifacts/experiments/<run>/perception/point_prompt/
    frame_<id>/result.json, masks/, images/, calls/, kept.jpg
    config.json, prompt.txt, summary.json
A run resumes where it stopped (finished frames are skipped) and refuses to continue if the
model, prompt, frames or vocabulary differ from what it started with.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src import config
from src.harness.client import Client, image as image_part

from ..io import perception_dir
from .clicks import POINT, SAM_CHECKPOINT, Frame, Sam, draw_grid, kept_overview, tile, tiles_sheet

PROMPT = config.PROMPT_ROOT / "point_objects_sep_v5.txt"
MODEL = "openai/gpt-6-sol"
MAX_ROUNDS = 24
SEGMENT_MAX = 6
FRAMES = config.ARTIFACTS_ROOT / "frames/scene0000_00"
FRAMES_FROM = config.ARTIFACTS_ROOT / "sources/scene0000_00/scene_amb3r_full100_v1/manifest.json"
OVERLAP_NOTE = ("fine if {inner} rests on or in {outer} (books on a shelf), "
                "wrong if {outer} leaked onto a separate object")


def fn(name, description, properties, required=None):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required or list(properties),
        "additionalProperties": False}}}


TOOLS = [
    fn("plan", "Declare objects to segment. Returns plan ids.", {"objects": {"type": "array", "minItems": 1, "items": {
        "type": "object", "additionalProperties": False, "required": ["label", "where"],
        "properties": {"label": {"type": "string", "description": "category name, English"},
                       "where": {"type": "string", "description": "where it is in the frame, a few words"}}}}}),
    fn("segment", "Run SAM on up to six planned objects. Each item: plan_id, positive points, optional negative points, "
                  "as [x, y] in full-image pixels. Returns one candidate per item, drawn as tiles.",
       {"items": {"type": "array", "minItems": 1, "maxItems": SEGMENT_MAX, "items": {
           "type": "object", "additionalProperties": False, "required": ["plan_id", "positive", "negative"],
           "properties": {"plan_id": {"type": "string"},
                          "positive": {"type": "array", "minItems": 1, "maxItems": 20, "items": POINT},
                          "negative": {"type": "array", "maxItems": 8, "items": POINT}}}}}),
    fn("zoom", "Magnified crop with a finer grid labelled in full-image pixels.",
       {"box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4,
                "description": "[x0, y0, x1, y1] in full-image pixels"}}),
    fn("combine", "Make one candidate from the union of several candidates of the same plan id, minus optional "
                  "candidates to subtract. Only for completing ONE unit that a single point set could not cover; "
                  "never for joining two adjacent units.",
       {"plan_id": {"type": "string"}, "union": {"type": "array", "minItems": 2, "items": {"type": "string"}},
        "subtract": {"type": "array", "items": {"type": "string"}},
        "why_one_unit": {"type": "string", "description": "what shows that the union pieces are parts of one unit "
                                                           "and not two units standing side by side"}}),
    fn("keep", "Accept candidates (C ids). A later keep for the same plan id replaces the earlier one.",
       {"candidates": {"type": "array", "minItems": 1, "items": {"type": "string"}}}),
    fn("finish", "End the frame. Every plan id must be kept or listed here with a reason.",
       {"skipped": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                "required": ["plan_id", "reason"],
                                                "properties": {"plan_id": {"type": "string"},
                                                               "reason": {"type": "string"}}}}}),
]


class ObjectFrame:
    """The tools of one object conversation, on one shared-code `Frame`."""

    def __init__(self, fid, sam, out: Path, frames_dir: Path = FRAMES, vocab=None, model=MODEL):
        self.fr = Frame(fid, sam, out, frames_dir, vocab, {"P": 0, "C": 0}, model, tag="")
        self.finished = False

    def first_images(self):
        fr = self.fr
        return [fr.save_img(fr.img, "frame"), fr.save_img(draw_grid(fr.img, 0, 0, fr.W, fr.H, 100, 1.0), "grid")]

    def t_plan(self, objects):
        fr = self.fr
        if fr.vocab:
            bad = sorted({o["label"].strip() for o in objects} - set(fr.vocab))
            if bad:
                raise ValueError(f"labels not in this run's vocabulary {fr.vocab}: {bad}; "
                                 f"the label must be exactly one of {fr.vocab}")
        ids = []
        for o in objects:
            pid = fr.plan_item(o["label"], o["where"])
            ids.append({"plan_id": pid, **fr.plan[pid]})
        return {"planned": ids, "total_planned": len(fr.plan)}, []

    def t_segment(self, items):
        fr = self.fr
        with fr.sam.lock:
            fr.sam.set_image(fr.img, fr.fid)
            rows, tiles = fr.segment_rows(items)
        p = fr.save_img(tiles_sheet(tiles), "segment")
        return {"candidates": rows,
                "image": "One tile per candidate, in order: mask tinted blue with cyan outline, rest dimmed; "
                         "green dots = positive points, red crosses = negative points."}, [p]

    def t_zoom(self, box):
        return self.fr.zoom(box)

    def t_combine(self, plan_id, union, subtract, why_one_unit):
        if not why_one_unit.strip():
            raise ValueError("why_one_unit is empty: say what shows the pieces are one unit")
        return self.fr.combine(plan_id, list(union), list(subtract), why_one_unit)

    def t_keep(self, candidates):
        fr = self.fr
        kept, overlaps = fr.keep(candidates, OVERLAP_NOTE)
        p = fr.save_img(kept_overview(fr.rgb, fr.kept), "kept")
        return {"kept": [f"{cid} -> {fr.cands[cid]['plan_id']}" for cid in kept], "kept_total": len(fr.kept),
                "planned_not_kept": [pid for pid in fr.plan if pid not in fr.kept], "overlap_warnings": overlaps,
                "image": "Every kept mask so far, labelled by plan id."}, [p]

    def t_finish(self, skipped):
        fr = self.fr
        sk = {s["plan_id"]: s["reason"] for s in skipped}
        bad = [pid for pid in sk if pid not in fr.plan]
        if bad:
            raise ValueError(f"unknown plan ids in skipped: {bad}")
        both = [pid for pid in sk if pid in fr.kept]
        if both:
            raise ValueError(f"{both} are kept; do not also skip them")
        missing = fr.unresolved(set(sk))
        if missing:
            raise ValueError(f"these plan ids are neither kept nor skipped: {missing}")
        fr.finish([{"plan_id": k, "reason": v} for k, v in sk.items()])
        self.finished = True
        return {"finished": True, "kept": len(fr.kept), "skipped": len(fr.skipped)}, []

    def call(self, name, args):
        if not any(t["function"]["name"] == name for t in TOOLS):
            raise ValueError(f"unknown tool {name!r}")
        return getattr(self, "t_" + name)(**args)


def run_frame(fid, sam, run_dir: Path, prompt, max_rounds, model=MODEL, effort=None, vocab=None, frames_dir=FRAMES):
    out = run_dir / f"frame_{fid:06d}"
    of = ObjectFrame(fid, sam, out, frames_dir, vocab, model)
    client = Client(model, out, timeout=240, reasoning_effort=effort)
    imgs = of.first_images()
    content = [{"type": "text", "text": f"Frame {fid}, {of.fr.W}x{of.fr.H} pixels. First the frame as captured, then "
                                        f"the same frame with a grid every 100 pixels (labels are x along the top, "
                                        f"y down the left)."},
               image_part(imgs[0]), image_part(imgs[1])]
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": content}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.monotonic()
    rounds = 0
    for step in range(max_rounds):
        rounds = step + 1
        if step == max_rounds - 2:
            messages.append({"role": "user", "content": "Two rounds left: keep what is good now, then call finish, "
                                                        "skipping what is not kept with a reason."})
        resp = client.call(messages, TOOLS, f"r{step:02d}")
        meta = sorted((out / "calls").glob(f"r{step:02d}_*_meta.json"))
        if meta:
            u = json.loads(meta[-1].read_text()).get("usage") or {}
            for k in usage:
                usage[k] += u.get(k) or 0
        messages.append(resp)
        call = resp["tool_calls"][0]
        name = call["function"]["name"]
        try:
            result, images = of.call(name, json.loads(call["function"]["arguments"]))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            result, images = {"error": str(exc)}, []
        print(f"  [{fid}] r{step:02d} {name}: {json.dumps(result, ensure_ascii=False)[:160]}", flush=True)
        (out / "calls" / f"r{step:02d}_tool.json").write_text(json.dumps(
            {"name": name, "arguments": call["function"]["arguments"], "result": result,
             "images": [str(p) for p in images]}, ensure_ascii=False, indent=2))
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
        for extra in resp["tool_calls"][1:]:             # parallel calls are off; answer any stray one
            messages.append({"role": "tool", "tool_call_id": extra["id"],
                             "content": json.dumps({"error": "one tool call per reply; this one was ignored"})})
        if images:
            messages.append({"role": "user", "content": [{"type": "text", "text": f"Image(s) from {name}:"}]
                             + [image_part(p) for p in images]})
        if of.finished:
            break
    of.fr.finished = of.finished
    return of.fr.write(rounds, usage, time.monotonic() - t0)


def one_frame(fid, sam, run_dir, prompt, max_rounds, model, effort, vocab):
    """Run one frame; a failure is recorded in error.json and does not stop the batch."""
    folder = run_dir / f"frame_{fid:06d}"
    try:
        r = run_frame(fid, sam, run_dir, prompt, max_rounds, model, effort, vocab)
        (folder / "error.json").unlink(missing_ok=True)
        print(f"DONE {fid}: finished={r['finished']} rounds={r['rounds']} kept={len(r['kept'])} "
              f"skipped={len(r['skipped'])} tokens={r['usage']} {r['seconds']}s", flush=True)
        return "finished" if r["finished"] else "unfinished"
    except Exception as exc:  # noqa: BLE001 - isolate the frame, keep the batch going
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "error.json").write_text(json.dumps(
            {"frame_id": fid, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
            ensure_ascii=False, indent=2))
        print(f"ERROR {fid}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        return "error"


def frames_from(path: Path) -> list[int]:
    return [int(f["source_frame_id"]) for f in json.loads(Path(path).read_text())["frames"]]


def run_dir_of(run: str) -> Path:
    return perception_dir(run) / "point_prompt"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="experiment name under artifacts/experiments")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--frames", type=int, nargs="+")
    pick.add_argument("--frames-from", type=Path, default=FRAMES_FROM, help="source manifest.json; its source_frame_ids")
    ap.add_argument("--prompt", type=Path, default=PROMPT)
    ap.add_argument("--vocab", help="comma-separated plan labels; any other label is refused")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--reasoning-effort", default=None, choices=["none", "low", "medium", "high"])
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--workers", type=int, default=4, help="frames in flight; SAM is shared under a lock")
    ap.add_argument("--sam-checkpoint", type=Path, default=SAM_CHECKPOINT)
    args = ap.parse_args(argv)
    frames = args.frames or frames_from(args.frames_from)
    out = run_dir_of(args.run)
    out.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text()
    vocab = [v.strip() for v in args.vocab.split(",")] if args.vocab else None
    cfg = {"model": args.model, "sam_checkpoint": str(args.sam_checkpoint), "max_rounds": args.max_rounds,
           "prompt_file": args.prompt.name, "vocab": vocab, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
           "frames_from": None if args.frames else str(args.frames_from), "frames": frames}
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        for key in ("model", "prompt_sha256", "frames", "max_rounds", "vocab"):
            if old.get(key) != cfg[key]:
                raise SystemExit(f"{key} differs from the existing run in {out}; choose a new --run")
    cfg_path.write_text(json.dumps({**cfg, "workers": args.workers}, indent=2))
    (out / "prompt.txt").write_text(prompt)
    todo = [fid for fid in frames if not (out / f"frame_{fid:06d}" / "result.json").exists()]
    print(f"{len(frames)} frames, {len(frames) - len(todo)} already done, {len(todo)} to run, "
          f"{args.workers} workers", flush=True)
    sam = Sam(args.sam_checkpoint)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda fid: one_frame(fid, sam, out, prompt, args.max_rounds, args.model,
                                            args.reasoning_effort, vocab), todo))
    summary = {"frames": len(frames), "finished": 0, "unfinished": 0, "error": [], "kept": 0, "skipped": 0,
               "rounds": 0, "prompt_tokens": 0, "completion_tokens": 0,
               "this_invocation_seconds": round(time.monotonic() - t0, 1)}
    for fid in frames:
        folder = out / f"frame_{fid:06d}"
        if (folder / "result.json").exists():
            r = json.loads((folder / "result.json").read_text())
            summary["finished" if r["finished"] else "unfinished"] += 1
            summary["kept"] += len(r["kept"]); summary["skipped"] += len(r["skipped"]); summary["rounds"] += r["rounds"]
            summary["prompt_tokens"] += r["usage"]["prompt_tokens"]
            summary["completion_tokens"] += r["usage"]["completion_tokens"]
        else:
            summary["error"].append(fid)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
