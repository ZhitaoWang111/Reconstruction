"""One frame of a segmentation conversation: SAM, the clicking tools and the images they return.

`Frame` holds one frame's plan, candidates and kept masks. Its tools (plan, segment,
combine, zoom, keep, finish) are what window.py exposes to the model, with a frame id
added to each call. Ids (P1, C1, ...) are counted across the whole window, so a plan id
or candidate id names exactly one frame.

Rules that came out of the scene0000_00 runs and are enforced here rather than in the
prompt, so the model gets a clear error instead of a silent bad mask:

* labels must be in the run's vocabulary (wall / floor / door / window);
* a mask below MIN_AREA_FRACTION of the frame is not kept: the lifting drops it anyway;
* `segment` reports positive points outside the mask, negative points inside it, masks
  in several pieces, and overlap with masks already kept in that frame.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from src import config

SAM_CHECKPOINT = config.WEIGHTS_ROOT / "sam3.pt"     # SAM3 with interactive (point) prediction
MIN_AREA_FRACTION = 0.005                             # the same floor as the text-prompt perception
PALETTE = [(255, 99, 71), (30, 144, 255), (50, 205, 50), (255, 193, 7), (186, 85, 211), (0, 206, 209),
           (255, 105, 180), (255, 140, 0), (70, 130, 180), (154, 205, 50), (220, 20, 60), (64, 224, 208),
           (138, 43, 226), (210, 180, 140), (0, 191, 255), (240, 128, 128)]
POINT = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}


def font(size: int):
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------- drawing
def draw_grid(img: Image.Image, x0, y0, x1, y1, step, scale) -> Image.Image:
    """Grid lines every `step` frame pixels over the crop [x0, x1) x [y0, y1) drawn at `scale`."""
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    f = font(15)
    for x in range(math.ceil(x0 / step) * step, x1 + 1, step):
        px = round((x - x0) * scale)
        d.line([(px, 0), (px, img.size[1])], fill=(255, 255, 0, 110), width=1)
        d.text((px + 2, 2), str(x), font=f, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    for y in range(math.ceil(y0 / step) * step, y1 + 1, step):
        py = round((y - y0) * scale)
        d.line([(0, py), (img.size[0], py)], fill=(255, 255, 0, 110), width=1)
        d.text((2, py + 2), str(y), font=f, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


def draw_points(d, pos, neg, mapf, r=7):
    for x, y in pos:
        px, py = mapf(x, y)
        d.ellipse([px - r, py - r, px + r, py + r], fill=(0, 230, 0), outline=(0, 0, 0), width=2)
    for x, y in neg:
        px, py = mapf(x, y)
        d.line([px - r, py - r, px + r, py + r], fill=(255, 0, 0), width=4)
        d.line([px - r, py + r, px + r, py - r], fill=(255, 0, 0), width=4)


def outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, iterations=2)


def tile(rgb, mask, pos, neg, title, size=(640, 478)) -> Image.Image:
    """One candidate for the model: the frame (or a crop round a small mask), mask tinted, points drawn."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) and (np.ptp(xs) < W / 4 and np.ptp(ys) < H / 4):
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        w = max(3 * (np.ptp(xs) + 1), 3 * (np.ptp(ys) + 1) * 4 / 3, 400)
        h = w * 3 / 4
        x0, y0 = int(max(0, min(W - w, cx - w / 2))), int(max(0, min(H - h, cy - h / 2)))
        x1, y1 = int(min(W, x0 + w)), int(min(H, y0 + h))
        title += "  [crop]"
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    a = rgb.astype(np.float32).copy()
    a[~mask] *= 0.45
    a[mask] = 0.55 * a[mask] + 0.45 * np.array([0, 160, 255])
    a[outline(mask)] = (0, 200, 255)
    crop = Image.fromarray(a[y0:y1, x0:x1].astype(np.uint8))
    sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)
    crop = crop.resize(size, Image.Resampling.BILINEAR)
    d = ImageDraw.Draw(crop)
    draw_points(d, pos, neg, lambda x, y: ((x - x0) * sx, (y - y0) * sy), r=6)
    d.rectangle([0, 0, size[0], 26], fill=(0, 0, 0))
    d.text((6, 3), title, font=font(18), fill=(255, 255, 255))
    return crop


def tiles_sheet(tiles, cols=3) -> Image.Image:
    cols = min(cols, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
    return sheet


def kept_overview(rgb, kept) -> Image.Image:
    a = rgb.astype(np.float32).copy()
    order = sorted(kept.values(), key=lambda k: -k["mask"].sum())        # big first so small ones stay visible
    for k in order:
        c = np.array(PALETTE[k["color"] % len(PALETTE)])
        a[k["mask"]] = 0.5 * a[k["mask"]] + 0.5 * c
    for k in order:
        a[outline(k["mask"])] = PALETTE[k["color"] % len(PALETTE)]
    img = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(img)
    for k in order:
        ys, xs = np.nonzero(k["mask"])
        d.text((int(xs.min()) + 4, int(ys.min()) + 4), f"{k['plan_id']} {k['label']}", font=font(18),
               fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    return img


def titled(img: Image.Image, title: str, size=(640, 478)) -> Image.Image:
    im = img.resize(size, Image.Resampling.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, size[0], 26], fill=(0, 0, 0))
    d.text((6, 3), title, font=font(18), fill=(255, 255, 255))
    return im


# ---------------------------------------------------------------- SAM
class Sam:
    """SAM3 interactive prediction behind a lock: windows run in threads, one model, one cached embedding."""

    def __init__(self, checkpoint: Path = SAM_CHECKPOINT):
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.torch = torch
        self.model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False,
                                            enable_inst_interactivity=True)
        self.proc = Sam3Processor(self.model)
        self.lock = threading.Lock()
        self.current = None

    def set_image(self, img, key):
        if self.current == key:
            return
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            self.state = self.proc.set_image(img)
        self.current = key

    def segment(self, pos, neg):
        """Points, then points plus the previous logits twice; a single click takes the best of three masks.
        On frame 2794 the same points gave 576 -> 160 -> 42 separate pieces over the three passes."""
        pts = np.array(pos + neg, dtype=np.float32)
        lab = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            m, s, lg = self.model.predict_inst(self.state, point_coords=pts, point_labels=lab,
                                               multimask_output=len(pts) == 1)
            first = float(s[int(np.argmax(s))])
            for _ in range(2):
                m, s, lg = self.model.predict_inst(self.state, point_coords=pts, point_labels=lab,
                                                   mask_input=lg[int(np.argmax(s))][None], multimask_output=False)
        return m[0].astype(bool), float(s[0]), first


# ---------------------------------------------------------------- one frame
class Frame:
    def __init__(self, fid: int, sam, out: Path, frames_dir: Path, vocab, ids: dict, model: str):
        self.fid, self.sam, self.out, self.vocab, self.ids, self.model = fid, sam, out, list(vocab), ids, model
        self.tag = f"frame {fid}  "
        self.img = Image.open(frames_dir / f"frame_{fid:06d}.jpg").convert("RGB")
        self.rgb = np.asarray(self.img)
        self.H, self.W = self.rgb.shape[:2]
        self.plan, self.cands, self.kept, self.skipped = {}, {}, {}, []
        self.finished = False
        self.n_img = 0
        for sub in ("images", "masks"):
            (out / sub).mkdir(parents=True, exist_ok=True)

    def save_img(self, img, name):
        self.n_img += 1
        p = self.out / "images" / f"{self.n_img:02d}_{name}.jpg"
        img.save(p, quality=90)
        return p

    def _new_id(self, kind):
        self.ids[kind] += 1
        return f"{kind}{self.ids[kind]}"

    def _pt(self, p):
        x, y = int(p[0]), int(p[1])
        if not (0 <= x < self.W and 0 <= y < self.H):
            raise ValueError(f"point {p} is outside the {self.W}x{self.H} image")
        return [x, y]

    def min_area(self) -> int:
        return int(MIN_AREA_FRACTION * self.W * self.H)

    # tools -------------------------------------------------------------
    def plan_item(self, label: str, where: str) -> str:
        label = label.strip()
        if label not in self.vocab:
            raise ValueError(f"label {label!r} is not allowed; the label must be exactly one of {self.vocab}")
        pid = self._new_id("P")
        self.plan[pid] = {"label": label, "where": where.strip()}
        return pid

    def segment_rows(self, items):
        """SAM on each item; one candidate row and one tile per item. Caller holds the SAM lock."""
        rows, tiles = [], []
        for it in items:
            pid = it["plan_id"]
            if pid not in self.plan:
                raise ValueError(f"{pid} is not a plan id; call plan first")
            pos = [self._pt(p) for p in it["positive"]]
            neg = [self._pt(p) for p in it.get("negative") or []]
            mask, score, first_score = self.sam.segment(pos, neg)
            cid = self._new_id("C")
            area = int(mask.sum())
            warnings = []
            miss = [p for p in pos if not mask[p[1], p[0]]]
            hit = [p for p in neg if mask[p[1], p[0]]]
            if miss:
                warnings.append(f"positive points outside the mask: {miss}")
            if hit:
                warnings.append(f"negative points inside the mask: {hit}")
            if area < self.min_area():
                warnings.append(f"area {area} px is below the downstream minimum ({self.min_area()} px); it would be dropped")
            lab, comps = ndimage.label(mask)
            big = [s for s in ndimage.sum(mask, lab, range(1, comps + 1)) if s > 0.02 * max(area, 1)] if comps else []
            if len(big) > 1:
                warnings.append(f"mask has {len(big)} separate pieces")
            for opid, k in self.kept.items():
                if opid == pid:
                    continue
                ov = int((mask & k["mask"]).sum())
                if area and ov / area > 0.10:
                    warnings.append(f"{ov / area:.0%} of this mask overlaps kept {opid} ({k['label']})")
            ys, xs = np.nonzero(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if area else None
            self.cands[cid] = {"plan_id": pid, "mask": mask, "pos": pos, "neg": neg, "score": score,
                               "first_pass_score": first_score, "bbox": bbox, "area": area}
            rows.append({"candidate": cid, "plan_id": pid, "label": self.plan[pid]["label"],
                         "area_fraction": round(area / (self.W * self.H), 4), "bbox_xyxy": bbox,
                         "sam_score": round(score, 3), "warnings": warnings})
            tiles.append(tile(self.rgb, mask, pos, neg, f"{self.tag}{cid}  {pid} {self.plan[pid]['label']}"))
        return rows, tiles

    def combine(self, plan_id, union, subtract, reason):
        if plan_id not in self.plan:
            raise ValueError(f"{plan_id} is not a plan id")
        for cid in list(union) + list(subtract):
            if cid not in self.cands:
                raise ValueError(f"{cid} is not a candidate id")
        wrong = [cid for cid in union if self.cands[cid]["plan_id"] != plan_id]
        if wrong:
            raise ValueError(f"{wrong} were segmented for other plan ids; union only parts of {plan_id}")
        mask = np.zeros_like(self.cands[union[0]]["mask"])
        for cid in union:
            mask |= self.cands[cid]["mask"]
        for cid in subtract:
            mask &= ~self.cands[cid]["mask"]
        pos = [p for cid in union for p in self.cands[cid]["pos"]]
        neg = [p for cid in union for p in self.cands[cid]["neg"]]
        cid = self._new_id("C")
        area = int(mask.sum())
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if area else None
        score = min(self.cands[c]["score"] for c in union)
        self.cands[cid] = {"plan_id": plan_id, "mask": mask, "pos": pos, "neg": neg, "score": score,
                           "first_pass_score": score, "bbox": bbox, "area": area, "combined_from": list(union),
                           "subtracted": list(subtract), "why_one_unit": reason}
        p = self.save_img(tile(self.rgb, mask, pos, neg, f"{self.tag}{cid}  {plan_id} {self.plan[plan_id]['label']} (combined)",
                               size=(1000, 747)), "combine")
        return {"candidate": cid, "plan_id": plan_id, "area_fraction": round(area / (self.W * self.H), 4),
                "bbox_xyxy": bbox, "combined_from": list(union), "subtracted": list(subtract)}, [p]

    def zoom(self, box):
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, x1 = sorted((max(0, x0), min(self.W, x1)))
        y0, y1 = sorted((max(0, y0), min(self.H, y1)))
        if x1 - x0 < 40 or y1 - y0 < 40:
            raise ValueError("zoom box must be at least 40x40 pixels inside the image")
        scale = 1000 / max(x1 - x0, y1 - y0)
        crop = self.img.crop((x0, y0, x1, y1)).resize((round((x1 - x0) * scale), round((y1 - y0) * scale)),
                                                      Image.Resampling.LANCZOS)
        step = next(s for s in (10, 20, 25, 50, 100) if max(x1 - x0, y1 - y0) / s <= 16)
        p = self.save_img(draw_grid(crop, x0, y0, x1, y1, step, scale), "zoom")
        return {"box": [x0, y0, x1, y1], "grid_step_px": step}, [p]

    def keep(self, candidates, overlap_note):
        done = []
        for cid in candidates:
            if cid not in self.cands:
                raise ValueError(f"{cid} is not a candidate id")
            c = self.cands[cid]
            pid = c["plan_id"]
            if c["area"] < self.min_area():
                raise ValueError(f"{cid} covers {c['area']} px, below the downstream minimum of {self.min_area()} px, "
                                 f"so it would be dropped; skip {pid} in finish instead, or segment it again if the "
                                 "surface is larger")
            color = self.kept[pid]["color"] if pid in self.kept else len(self.kept)
            self.kept[pid] = {**c, "candidate": cid, "label": self.plan[pid]["label"], "color": color}
            done.append(cid)
        overlaps = []
        ks = list(self.kept.items())
        for i, (pa, ka) in enumerate(ks):
            for pb, kb in ks[i + 1:]:
                ov = int((ka["mask"] & kb["mask"]).sum())
                small = min(ka["area"], kb["area"])
                if small and ov / small > 0.3:
                    inner, outer = (pa, pb) if ka["area"] <= kb["area"] else (pb, pa)
                    overlaps.append(f"{ov / small:.0%} of {inner} ({self.kept[inner]['label']}) lies inside "
                                    f"{outer} ({self.kept[outer]['label']}): " + overlap_note.format(inner=inner, outer=outer))
        return done, overlaps

    def finish(self, skipped):
        self.skipped = [{"plan_id": s["plan_id"], "label": self.plan[s["plan_id"]]["label"], "reason": s["reason"]}
                        for s in skipped]
        self.finished = True

    def unresolved(self, skipped_ids) -> list:
        return [(pid, self.plan[pid]["label"]) for pid in self.plan if pid not in self.kept and pid not in skipped_ids]

    def write(self, rounds, usage, seconds, extra=None) -> dict:
        """result.json and the kept masks: what src.structure.lift reads."""
        kept = []
        for i, (pid, k) in enumerate(sorted(self.kept.items(), key=lambda kv: int(kv[0][1:]))):
            f = f"masks/{i + 1:03d}_{pid}.png"
            Image.fromarray(k["mask"].astype(np.uint8) * 255).save(self.out / f)
            kept.append({"plan_id": pid, "label": k["label"], "where": self.plan[pid]["where"],
                         "candidate": k["candidate"], "positive": k["pos"], "negative": k["neg"],
                         "sam_score": k["score"], "first_pass_score": k["first_pass_score"],
                         "bbox_xyxy": k["bbox"], "mask_pixels": k["area"],
                         "combined_from": k.get("combined_from"), "subtracted": k.get("subtracted"),
                         "why_one_unit": k.get("why_one_unit"),
                         "area_fraction": k["area"] / (self.W * self.H), "mask_file": f})
        if self.kept:
            kept_overview(self.rgb, self.kept).save(self.out / "kept.jpg", quality=92)
        result = {"frame_id": self.fid, "finished": self.finished, "rounds": rounds, "seconds": round(seconds, 1),
                  "usage": usage, "plan": self.plan, "kept": kept, "skipped": self.skipped,
                  "segment_calls": len(self.cands), "model": self.model, **(extra or {})}
        (self.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
