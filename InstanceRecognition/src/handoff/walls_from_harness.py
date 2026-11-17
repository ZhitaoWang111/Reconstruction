"""Write the wall handoff (walls/v1) from the wall geometry of a structure segmentation run.

    python -m src.handoff.walls_from_harness --run struct_v4_win5

Inputs, all produced earlier by `src.structure` and read as files:

* `artifacts/structure/<run>/closed.json` - the closed walls: one storey height, thickness,
  the outline rectangle, each wall's observed and inferred pieces;
* `artifacts/structure/<run>/walls.json` - the observed wall planes they came from, and the
  floor plane that defines the floor frame;
* the run's kept 2D wall masks (one result.json per segmented frame);
* the AMB3R point maps, poses and the Sim(3) into the scan frame, as `src.structure.lift` used them.

Two facts that shaped this file and must not be rediscovered downstream:

1. Key-frame camera poses are AMB3R's own poses, Sim(3)-corrected into the scan frame,
   not the ScanNet GT poses. The wall geometry was lifted with AMB3R, so its own poses
   are the ones that put a wall back where its pixels are: projecting the lifted wall
   points back into frames 112, 894, 1229 and 4471 lands a median 6-16 px from their
   pixel with AMB3R poses and 24-79 px with GT poses.
2. A 2D wall mask was not split at corners (the prompt said so on purpose), so one
   mask can show two walls. A wall's key-frame mask is cut from the frame's wall masks:
   it keeps the pixels whose lifted point lies on that wall's plane, at the depth that
   frame saw the plane (one frame can sit up to ~0.1 m off another; see `plane_offset_m`).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from src import config
from src.geometry.amb3r_mask_source import AMBR3R_INTERMEDIATE, AMBR3R_SIZE, DEFAULT_NPZ, DEFAULT_SIM3
from src.geometry.backproject_masks import align_mask, crop_box
from src.structure.frames import FRAMES, MASK_RUNS, OUT_ROOT, SCENE, FloorFrame, PointMaps, frame_slots
from src.structure.lift import grid_normals

from . import schema, walls_schema as ws
from .from_gt import BOX_COLOUR, frame_clarity, read_scene_meta

DEFAULT_OUTPUT = config.ARTIFACTS_ROOT / "handoff/scene0000_00/walls_harness"
OFFSET_SEARCH = 0.30       # m; points this close to the plane estimate the frame's depth for it
OFFSET_KEEP = 0.10         # m; pixels kept within this of that depth
ALONG_MARGIN = 0.30        # m; a mask pixel may lie this far past the observed extent of its plane
MIN_CELLS = 150            # grid cells of one wall in one frame worth considering
MIN_AREA = 0.25            # m2 of visible wall for a candidate key frame (a strip above a counter is not a view of the wall)
MAX_VIEW_DEG = 75.0        # steeper views are not key frames
COVER_BIN = 0.10           # m; wall-surface cells (along x height) for the coverage-greedy choice
STOP_NEW_M2 = 0.15         # after the first frame: stop when the best frame adds less new wall surface than this
OUTLINE_PX = 3             # the wall outline in boxed.jpg: a ring this wide just outside the mask
RECT_MAX_DEG = 60.0        # rectified views only below this view angle
RECT_PX_PER_M = 150
FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_LATIN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ---------------------------------------------------------------- frames and geometry
class Frames:
    """Floor frame <-> scan frame, and the AMB3R point map of one frame in the floor frame."""

    def __init__(self, walls_json: dict, maps: PointMaps | None = None):
        self.maps = maps or PointMaps()
        self.floor = FloorFrame(walls_json["floor"])
        A = self.maps.A
        floor_to_lift = np.eye(4); floor_to_lift[:3, :3] = self.floor.R.T; floor_to_lift[:3, 3] = self.floor.c
        self.floor_to_scan = np.linalg.inv(A) @ floor_to_lift

    def to_floor(self, p_scan):
        A = self.maps.A
        return self.floor(np.asarray(p_scan, dtype=np.float64) @ A[:3, :3].T + A[:3, 3])

    def to_scan(self, p_floor):
        p = np.asarray(p_floor, dtype=np.float64)
        return p @ self.floor_to_scan[:3, :3].T + self.floor_to_scan[:3, 3]

    def frame(self, seq: int):
        W, H = AMBR3R_SIZE
        p_scan = self.maps.scan(seq)
        valid = self.maps.valid(seq, p_scan)
        pose = self.maps.camera_to_world_scan(seq)
        P = self.to_floor(p_scan).reshape(H, W, 3)
        cam = self.to_floor(pose[:3, 3][None])[0]
        return P, valid.reshape(H, W), pose, p_scan.reshape(H, W, 3), cam


def grid_index(width: int, height: int):
    """For every original pixel, the AMB3R grid cell it falls in (the inverse of align_mask)."""
    gw, gh = AMBR3R_SIZE
    iw, ih = AMBR3R_INTERMEDIATE
    left, top, right, bottom = crop_box(AMBR3R_INTERMEDIATE, AMBR3R_SIZE)
    x = (np.arange(width) + 0.5) * iw / width
    y = (np.arange(height) + 0.5) * ih / height
    u = np.clip(np.floor((x - left) * gw / (right - left)).astype(int), 0, gw - 1)
    v = np.clip(np.floor((y - top) * gh / (bottom - top)).astype(int), 0, gh - 1)
    return u, v


def grid_pixel_centres(width: int, height: int):
    """The original-pixel position of every grid cell's centre, (H, W) arrays."""
    gw, gh = AMBR3R_SIZE
    iw, ih = AMBR3R_INTERMEDIATE
    left, top, right, bottom = crop_box(AMBR3R_INTERMEDIATE, AMBR3R_SIZE)
    u, v = np.meshgrid(np.arange(gw), np.arange(gh))
    x = (left + (u + 0.5) * (right - left) / gw) * width / iw - 0.5
    y = (top + (v + 0.5) * (bottom - top) / gh) * height / ih - 0.5
    return x, y


def line_intersection(p, d, q, e):
    """Point where p + s d meets q + t e, or None if they are within 5 degrees of parallel."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < np.sin(np.radians(5)):
        return None
    w = q - p
    s = (w[0] * e[1] - w[1] * e[0]) / den
    return p + s * d


# ---------------------------------------------------------------- the walls
def build_walls(closed: dict, planes: dict):
    """walls/v1 wall records (geometry, segments, joins, evidence) from the closed walls."""
    names = {0: "ext_W", 1: "ext_S", 2: "ext_E", 3: "ext_N"}
    side_of = {f"side{k}": v for k, v in names.items()}
    interior = [f for f in closed["final"] if f["role"] == "interior"]
    int_name = {f["walls"][0]: f"int_{i + 1}" for i, f in enumerate(sorted(interior, key=lambda f: f["walls"][0]))}
    src = {w["id"]: w for w in closed["walls"]}
    recs = []
    for f in closed["final"]:
        wid = names[f["side"]] if f["role"] == "exterior" else int_name[f["walls"][0]]
        p0 = np.asarray(f["pieces"][0]["p0"], dtype=np.float64)
        p1 = np.asarray(f["pieces"][-1]["p1"], dtype=np.float64)
        room = np.asarray(f["normal"], dtype=np.float64)
        room = room / np.linalg.norm(room)
        pieces = f["pieces"]
        d = p1 - p0
        flip = (np.array([-d[1], d[0]]) @ room) < 0          # one convention for every wall: the room on the left of start->end
        if flip:
            p0, p1 = p1, p0
            pieces = [{**pc, "p0": pc["p1"], "p1": pc["p0"]} for pc in reversed(pieces)]
        recs.append({"id": wid, "role": f["role"], "start": p0, "end": p1, "room": room,
                     "t": float(f["thickness_m"]), "pieces": pieces, "planes": list(f["walls"]),
                     "hint": f.get("thickness_hint", []), "flip": bool(flip)})
    by_id = {r["id"]: r for r in recs}
    # joins: exterior corners mitred around the rectangle; interior ends from close.py's extension step
    order = ["ext_W", "ext_S", "ext_E", "ext_N"]
    for i, wid in enumerate(order):
        by_id[wid]["joins"] = {"start": {"wall": order[i - 1], "type": "mitre"},
                               "end": {"wall": order[(i + 1) % 4], "type": "mitre"}}
    for f in interior:
        w = src[f["walls"][0]]
        j = {}
        flip = by_id[int_name[f["walls"][0]]]["flip"]
        for e in w["ends"]:
            side = {"start": "end", "end": "start"}[e["side"]] if flip else e["side"]
            other = e.get("meets")
            if other is None or (e["ext_m"] == 0.0 and e.get("why")):
                j[side] = {"wall": None, "type": "none"}
            elif other.startswith("side"):
                j[side] = {"wall": side_of[other], "type": "butt"}
            else:
                j[side] = {"wall": int_name[other], "type": "mitre"}
        by_id[int_name[f["walls"][0]]]["joins"] = j

    def lines(r):
        d = (r["end"] - r["start"]) / np.linalg.norm(r["end"] - r["start"])
        back = -r["room"]
        return d, back

    # corners: every end snapped onto the line it joins, so the outline is exactly closed
    for r in recs:
        d, back = lines(r)
        ends = {}
        for side in ("start", "end"):
            p = r[side]
            jn = r["joins"][side]
            P, Q = p, p + r["t"] * back
            if jn["type"] != "none":
                o = by_id[jn["wall"]]; od, oback = lines(o)
                hit = line_intersection(p, d, o["start"], od)
                if hit is not None:
                    P = hit
                    if jn["type"] == "mitre":
                        Q = line_intersection(p + r["t"] * back, d, o["start"] + o["t"] * oback, od)
                    else:                           # butt: the back face stops at the other wall's room face
                        Q = line_intersection(p + r["t"] * back, d, o["start"], od)
                    if Q is None:
                        Q = P + r["t"] * back
            ends[side] = (P, Q)
        r["P0"], r["Q0"] = ends["start"]; r["P1"], r["Q1"] = ends["end"]

    out = []
    for r in recs:
        d0 = r["P1"] - r["P0"]; L = float(np.linalg.norm(d0)); d = d0 / L
        back = -r["room"]
        back = back - (back @ d) * d; back /= np.linalg.norm(back)      # exactly perpendicular to the line
        # segments: piece boundaries projected on the snapped line, clipped, contiguous
        cuts, kinds = [0.0], []
        for k, pc in enumerate(r["pieces"]):
            to = L if k == len(r["pieces"]) - 1 else float(np.clip((np.asarray(pc["p1"]) - r["P0"]) @ d, 0, L))
            if to - cuts[-1] < 1e-4:
                continue
            cuts.append(to); kinds.append(pc["kind"])
        segs = [{"from_m": round(cuts[i], 4), "to_m": round(cuts[i + 1], 4), "kind": kinds[i],
                 "source_planes": r["planes"] if kinds[i] == "observed" else []} for i in range(len(kinds))]
        segs[-1]["to_m"] = round(L, 4)
        ev_planes = [src[p] for p in r["planes"]]
        detected = []
        for pl in ev_planes:
            for o in pl.get("openings", []) or []:
                ends = [np.asarray(c[:2]) for c in o["corners"][:2]]
                along = sorted(float((e - r["P0"]) @ d) for e in ends)
                detected.append({"from_m": round(along[0], 3), "to_m": round(along[1], 3),
                                 "z_floor": [round(o["z"][0], 3), round(o["z"][1], 3)],
                                 "width_m": round(o["width_m"], 3), "height_m": round(o["height_m"], 3),
                                 "rays_through": int(o["rays"]), "closed_above": bool(o["closed_above"]),
                                 "note": "found by the ray test (rays pass through, solid above); "
                                         "not built in walls/v1"})
        out.append({
            "id": r["id"], "role": r["role"],
            "start_floor": [round(float(x), 5) for x in r["P0"]], "end_floor": [round(float(x), 5) for x in r["P1"]],
            "length_m": round(L, 4), "normal_to_back": [round(float(x), 6) for x in back],
            "thickness_m": r["t"], "thickness_kind": "assumed",
            "height_m": float(closed["storey"]["height_m"]),
            "footprint_floor": [[round(float(x), 5) for x in q] for q in (r["P0"], r["P1"], r["Q1"], r["Q0"])],
            "segments": segs, "joins": r["joins"], "openings": [], "key_frames": [],
            "evidence": {
                "planes": r["planes"],
                "frames": sorted({int(x) for pl in ev_planes for x in pl["frames"]}),
                "masks": sorted({int(x) for pl in ev_planes for x in pl["masks"]}),
                "observed_m": round(sum(s["to_m"] - s["from_m"] for s in segs if s["kind"] == "observed"), 3),
                "inferred_m": round(sum(s["to_m"] - s["from_m"] for s in segs if s["kind"] == "inferred"), 3),
                "plane_rms_cm": [round(pl["rms_cm"], 2) for pl in ev_planes],
                "snap_deg": [round(pl.get("snap_deg", 0.0), 2) for pl in ev_planes],
                "behind_seen": [round(pl["behind_seen"], 3) for pl in ev_planes],
                "observed_top_m": [round(pl["z"][1], 3) for pl in ev_planes],
                "thickness_hint": r["hint"],
                "detected_openings": detected,
            },
            "_planes": ev_planes, "_dir": d,
        })
    return out


# ---------------------------------------------------------------- key frames
def frame_candidates(walls, frames: Frames, run_dir: Path, slot: dict, width: int, height: int, H: float):
    """For every segmented frame and every wall: the wall's own mask and how the frame sees it."""
    gu, gv = grid_index(width, height)
    gx, gy = grid_pixel_centres(width, height)
    cand = {w["id"]: [] for w in walls}
    for folder in sorted(run_dir.glob("frame_*")):
        r = json.loads((folder / "result.json").read_text())
        fid = int(r["frame_id"]); seq = slot[fid]
        wall_masks = [k for k in r["kept"] if k["label"] == "wall"]
        if not wall_masks:
            continue
        union = np.zeros((height, width), bool)
        for k in wall_masks:
            union |= np.asarray(Image.open(folder / k["mask_file"]).convert("L")) > 0
        ug = align_mask(union, AMBR3R_INTERMEDIATE, AMBR3R_SIZE)
        P, valid, pose, p_scan, cam = frames.frame(seq)
        N = grid_normals(P, cam)
        okn = np.isfinite(N).all(2)
        for w in walls:
            keep = np.zeros(ug.shape, bool); offsets = {}
            for pl in w["_planes"]:
                c, n, u = (np.asarray(pl[k], dtype=np.float64) for k in ("point", "normal", "along"))
                d = (P[..., :2] - c) @ n; t = (P[..., :2] - c) @ u
                base = ug & valid & (t > pl["a"][0] - ALONG_MARGIN) & (t < pl["a"][1] + ALONG_MARGIN) \
                    & (P[..., 2] > -0.1) & (P[..., 2] < H + 0.1)
                facing = okn & ((N[..., :2] @ n) > np.cos(np.radians(45)))
                est = base & facing & (np.abs(d) < OFFSET_SEARCH)
                if est.sum() < MIN_CELLS:
                    continue
                off = float(np.median(d[est]))
                k_ = base & (np.abs(d - off) < OFFSET_KEEP)
                k_ = ndimage.binary_closing(k_, np.ones((3, 3), bool), iterations=2) & ug
                if k_.sum() < MIN_CELLS:
                    continue
                keep |= k_; offsets[pl["id"]] = off
            if keep.sum() < MIN_CELLS:
                continue
            pts = P[keep & valid]
            area = len(np.unique(np.floor(pts / 0.03).astype(np.int64), axis=0)) * 0.03 ** 2
            centroid = pts.mean(0)
            v = centroid[:2] - cam[:2]
            dist = float(np.linalg.norm(v))
            room = -np.asarray(w["normal_to_back"])
            view = float(np.degrees(np.arccos(np.clip(-(v / max(dist, 1e-9)) @ room, -1, 1))))
            if area < MIN_AREA or view > MAX_VIEW_DEG:
                continue
            p0 = np.asarray(w["start_floor"]); d_ = w["_dir"]
            along = (pts[:, :2] - p0) @ d_
            cover = [float(np.clip(np.percentile(along, 2), 0, w["length_m"])),
                     float(np.clip(np.percentile(along, 98), 0, w["length_m"]))]
            # how well pose + intrinsics put the lifted points back on their own pixels
            Tinv = np.linalg.inv(pose)
            q = p_scan[keep & valid] @ Tinv[:3, :3].T + Tinv[:3, 3]
            K = frames.K
            u_ = K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2]; v_ = K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]
            reproj = float(np.median(np.hypot(u_ - gx[keep & valid], v_ - gy[keep & valid])))
            full = union & keep[gv[:, None], gu[None, :]]
            # wall surface this frame shows, as cells along the wall and up its height: a thin strip
            # a metre long covers little surface, a full-height view of the same metre covers a lot
            nb, nz = int(np.ceil(w["length_m"] / COVER_BIN)) + 1, int(np.ceil(H / COVER_BIN)) + 1
            bins = np.zeros((nb, nz), bool)
            bins[np.clip((along / COVER_BIN).astype(int), 0, nb - 1), np.clip((pts[:, 2] / COVER_BIN).astype(int), 0, nz - 1)] = True
            cand[w["id"]].append({"fid": fid, "seq": seq, "mask": full, "area": area, "view": view, "dist": dist,
                                  "cover": cover, "bins": bins, "pose": pose, "offsets": offsets,
                                  "reproj": reproj, "cam": cam})
    return cand


def choose(cands):
    """Coverage-greedy: each pick adds the most wall surface not yet shown, seen most squarely."""
    chosen, taken, covered = [], set(), None
    for _ in range(ws.MAX_KEYFRAMES):
        best = None
        for i, c in enumerate(cands):
            if i in taken:
                continue
            cov = c["bins"] if covered is None else c["bins"] & ~covered
            new = cov.sum() * COVER_BIN ** 2
            score = new * np.cos(np.radians(c["view"]))
            if best is None or score > best[0]:
                best = (score, c, new, i)
        if best is None or (chosen and best[2] < STOP_NEW_M2):
            break
        chosen.append(best[1]); taken.add(best[3])
        covered = best[1]["bins"] if covered is None else covered | best[1]["bins"]
    return chosen, (None if covered is None else covered.any(1))


def draw_boxed(rgb: Image.Image, mask: np.ndarray, label: str) -> Image.Image:
    """The frame as it is, with the wall's outline drawn just outside the mask and a label.

    Not one pixel of the wall is changed: the downstream module judges the wall's colour and
    texture from this picture, so nothing is dimmed, tinted or drawn over the wall. The
    outline is a ring outside the mask, and the label goes in the image corner that covers
    the least wall.
    """
    a = np.asarray(rgb).copy()
    ring = ndimage.binary_dilation(mask, iterations=OUTLINE_PX) & ~mask
    a[ring] = BOX_COLOUR
    image = Image.fromarray(a)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_LATIN, max(17, image.width // 45))
    except OSError:
        font = ImageFont.load_default()
    tb = draw.textbbox((0, 0), label, font=font)
    w, h = tb[2] - tb[0] + 8, tb[3] - tb[1] + 8
    corners = [(0, 0), (image.width - w, 0), (0, image.height - h), (image.width - w, image.height - h)]
    x, y = min(corners, key=lambda c: mask[c[1]:c[1] + h, c[0]:c[0] + w].sum())
    draw.rectangle((x, y, x + w - 1, y + h - 1), fill=(0, 0, 0))
    draw.text((x + 4 - tb[0], y + 4 - tb[1]), label, font=font, fill=(255, 255, 255))
    return image


def rectify(wall, kf_cand, frames: Frames, rgb: Image.Image, H: float):
    """This frame's view of the wall, re-sampled square-on: x = metres along the wall, y = height.

    Samples the image where this frame saw the wall (its own plane offset), placed at the
    wall line's coordinates. Alpha is the wall's mask: furniture and the rest stay transparent.
    """
    p0 = np.asarray(wall["start_floor"]); d = wall["_dir"]; room = -np.asarray(wall["normal_to_back"])
    s0 = max(0.0, kf_cand["cover"][0] - 0.1); s1 = min(wall["length_m"], kf_cand["cover"][1] + 0.1)
    nx = max(2, int(round((s1 - s0) * RECT_PX_PER_M))); ny = max(2, int(round(H * RECT_PX_PER_M)))
    s = s0 + (np.arange(nx) + 0.5) / RECT_PX_PER_M
    h = H - (np.arange(ny) + 0.5) / RECT_PX_PER_M
    q = p0 + s[:, None] * d                                      # on the wall line
    lam = np.zeros(nx)
    for pl in wall["_planes"]:                                   # move onto the plane this frame saw
        if pl["id"] in kf_cand["offsets"]:
            c, n = np.asarray(pl["point"]), np.asarray(pl["normal"])
            lam = (kf_cand["offsets"][pl["id"]] - (q - c) @ n) / max(room @ n, 1e-6)
            break
    xy = q + lam[:, None] * room
    X = np.stack(np.broadcast_arrays(xy[None, :, 0], xy[None, :, 1], h[:, None]), -1).reshape(-1, 3)
    p = frames.to_scan(X)
    Tinv = np.linalg.inv(kf_cand["pose"])
    pc = p @ Tinv[:3, :3].T + Tinv[:3, 3]
    K = frames.K
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]; v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    W_, H_ = rgb.size
    inside = (pc[:, 2] > 0.05) & (u >= 0) & (u < W_ - 1) & (v >= 0) & (v < H_ - 1)
    img = np.asarray(rgb).astype(np.float32)
    out = np.zeros((len(p), 4), np.uint8)
    ui, vi = u[inside], v[inside]
    x0, y0 = np.floor(ui).astype(int), np.floor(vi).astype(int); fx_, fy_ = ui - x0, vi - y0
    col = (img[y0, x0] * ((1 - fx_) * (1 - fy_))[:, None] + img[y0, x0 + 1] * (fx_ * (1 - fy_))[:, None]
           + img[y0 + 1, x0] * ((1 - fx_) * fy_)[:, None] + img[y0 + 1, x0 + 1] * (fx_ * fy_)[:, None])
    out[inside, :3] = np.clip(col, 0, 255).astype(np.uint8)
    out[inside, 3] = np.where(kf_cand["mask"][np.round(vi).astype(int), np.round(ui).astype(int)], 255, 0)
    image = Image.fromarray(out.reshape(ny, nx, 4), "RGBA")
    return image, [round(s0, 4), round(s0 + nx / RECT_PX_PER_M, 4)], [0.0, round(ny / RECT_PX_PER_M, 4)]


# ---------------------------------------------------------------- preview and README
def preview(doc: dict, frames_xy: dict, path: Path):
    walls = doc["walls"]
    pts = np.array([q for w in walls for q in w["footprint_floor"]] + [c for c in frames_xy.values()])
    lo, hi = pts.min(0) - 0.6, pts.max(0) + 0.6
    k = 130
    W, H = int((hi[0] - lo[0]) * k) + 340, int((hi[1] - lo[1]) * k) + 60
    img = Image.new("RGB", (W, H), (252, 252, 251)); dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_CJK, 15); small = ImageFont.truetype(FONT_CJK, 12)
    except OSError:
        font = small = ImageFont.load_default()
    S = lambda p: (30 + (p[0] - lo[0]) * k, H - 30 - (p[1] - lo[1]) * k)
    ext, intr, inf_col = (22, 25, 29), (98, 80, 214), (140, 144, 150)
    for w in walls:
        dr.polygon([S(q) for q in w["footprint_floor"]], fill=(205, 207, 211) if w["role"] == "exterior" else (210, 204, 244),
                   outline=ext if w["role"] == "exterior" else intr)
    for w in walls:
        p0, d = np.asarray(w["start_floor"]), np.asarray(w["end_floor"]) - np.asarray(w["start_floor"])
        d = d / np.linalg.norm(d)
        for sgm in w["segments"]:
            a, b = p0 + sgm["from_m"] * d, p0 + sgm["to_m"] * d
            col = ext if w["role"] == "exterior" else intr
            if sgm["kind"] == "observed":
                dr.line([S(a), S(b)], fill=col, width=4)
            else:
                n = max(1, int(np.linalg.norm(b - a) / 0.12))
                for i in range(0, n, 2):
                    dr.line([S(a + (b - a) * i / n), S(a + (b - a) * min(i + 1, n) / n)], fill=inf_col, width=3)
    colours = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164), (0, 131, 0), (98, 80, 214)]
    for i, w in enumerate(walls):
        col = colours[i % len(colours)]
        p0, d = np.asarray(w["start_floor"]), np.asarray(w["end_floor"]) - np.asarray(w["start_floor"])
        d = d / np.linalg.norm(d)
        for kf in w["key_frames"]:
            c = frames_xy[(w["id"], kf["source_frame_id"])]
            m = p0 + 0.5 * (kf["covers_m"][0] + kf["covers_m"][1]) * d
            dr.line([S(c), S(m)], fill=col, width=1)
            x, y = S(c); dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
        mid = p0 + 0.5 * w["length_m"] * d - np.asarray(w["normal_to_back"]) * 0.45
        label = f"{w['id']}  {int(round(w['thickness_m'] * 100))} cm"
        tb = dr.textbbox(S(mid), label, font=font, anchor="mm")
        dr.rectangle([tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=(255, 255, 255), outline=col)
        dr.text(S(mid), label, font=font, fill=(22, 25, 29), anchor="mm")
    lx = W - 300
    rows = [((205, 207, 211), "外墙占地（厚度假定）"), ((210, 204, 244), "内墙占地（厚度假定）"),
            (ext, "观测到的墙段"), (inf_col, "推断补上的墙段"), ((42, 120, 214), "关键帧相机 → 它拍到的墙段（颜色同墙的标签框）")]
    for i, (c, t) in enumerate(rows):
        y = 30 + i * 24
        dr.rectangle([lx, y, lx + 18, y + 12], fill=c); dr.text((lx + 26, y - 3), t, font=small, fill=(69, 76, 85))
    y = 30 + len(rows) * 24 + 10
    dr.text((lx, y), f"层高 {doc['storey']['height_m']:.2f} m（假定）", font=small, fill=(69, 76, 85))
    dr.text((lx, y + 20), f"{doc['outline']['inner_size_m'][0]:.2f} × {doc['outline']['inner_size_m'][1]:.2f} m 内净", font=small, fill=(69, 76, 85))
    bx, by = 30, H - 14
    dr.line([(bx, by), (bx + k, by)], fill=(69, 76, 85), width=3); dr.text((bx + k + 6, by - 9), "1 m", font=small, fill=(69, 76, 85))
    dr.text((W - 150, H - 24), "↑ +y（地面坐标系）", font=small, fill=(107, 116, 128))
    img.save(path)


README = """# 墙体交接文件（walls/v1）

场景 `{scene}`，来源 run `{run}`。只交付**墙体**：门、窗、开口、天花板、地面网格都不在里面（见 `not_included`）。
校验：`python -m src.handoff.walls_check {out}`。字段定义写在 `src/handoff/walls_schema.py` 里。

## 文件

| 文件 | 内容 |
|---|---|
| `walls.json` | 全部数据：层高、厚度、外墙轮廓、每面墙的几何、观测/推断分段、连接方式、关键帧 |
| `keyframes/<墙 id>/<seq>_rgb.jpg` | 关键帧原图（{w}×{h}） |
| `keyframes/<墙 id>/<seq>_mask.png` | 这面墙在这一帧里露出来的部分（0/255），已经按墙切开，不含家具、门窗和别的墙 |
| `keyframes/<墙 id>/<seq>_boxed.jpg` | 预览：原图不做任何调色，只在墙 mask **外侧**画一圈黄线标出这面墙，标签放在遮挡墙面最少的角上。墙面像素只差 JPEG 重新压缩带来的几个灰度级 |
| `preview.png` | 俯视图：占地、观测段（实线）、推断段（灰虚线）、关键帧相机位置 |

## 坐标

- 和物体交接（handoff/v2）一样：ScanNet **扫描坐标系**，米，+Z 向上，没有应用 axisAlignment。墙和物体可以直接叠放。
- 墙的几何写在**地面坐标系**里（地面在 z = 0、墙竖直），字段名带 `_floor`。用 `floor_frame_to_scan` 转到扫描坐标系：`p_scan = floor_frame_to_scan @ [x, y, z, 1]`。二维点 `[x, y]` 就是 z = 0 时的地面坐标。
- 关键帧的 `camera_to_world` 在扫描坐标系里，OpenCV 约定（+X 右、+Y 下、+Z 前），和物体交接相同，内参在 `camera`。用法：`p_cam = inverse(camera_to_world) @ p_scan`，`u = fx·x/z + cx`，`v = fy·y/z + cy`。

## 一面墙怎么生成

1. 墙线是 `start_floor → end_floor`，这是房间一侧的墙面。房间在墙线的左手边（沿 start→end 走时），四面外墙按逆时针排列。
2. 厚度朝 `normal_to_back` 加 `thickness_m`；房间一侧的墙面位置不变。
3. 从 z = 0 拉到 `height_m`。
4. `footprint_floor` 是现成的占地四边形，顺序为 `[start, end, end 背后, start 背后]`，转角已按 `joins` 处理：
   - `mitre`：两面墙在转角斜接（外墙四角，以及内墙之间）；
   - `butt`：内墙停在外墙的房间一侧墙面上，外墙不受影响。
   如果改了厚度，请用墙线和 `joins` 重算，不要直接改 `footprint_floor`。
5. `openings` 这一版都是空的，墙都是实心的。`evidence.detected_openings` 里记录了射线检验发现的门洞，没有用来生成。

## 外观：关键帧

- **判断墙的外观请用 `rgb.jpg` + `mask.png`**：`rgb.jpg` 是原始帧的逐字节拷贝，`mask.png` 标出这面墙露出来的像素。`boxed.jpg` 只是帮人认出是哪面墙的预览，经过一次 JPEG 重新压缩。
- 每面墙最多 3 帧，`role` 依次为 primary / secondary / third。选法是贪心：每次挑新拍到的墙段最多、拍得最正的一帧，所以三帧通常拍的是墙的不同段。
- `covers_m` 是这一帧拍到的沿墙范围（从 start 起算的米数）。推断补上的墙段没有任何一帧拍到；生成外观时请参考同一面墙已观测部分的外观。
- `view_angle_deg` 是视线和墙面法向的夹角，0° 表示正对。斜视超过 60° 的帧，墙面会被拉伸得厉害。
- `plane_offset_m` 是这一帧看到的墙面相对墙平面的深度偏差（不同帧之间可差约 0.1 m）；`reprojection_px` 是把这一帧的墙点投回图像时，和它原像素的距离中位数，用来衡量位姿和内参是否可信。
- 相机位姿是 AMB3R 自己的位姿（经 Sim(3) 转到扫描坐标系），不是 ScanNet 的 GT 位姿：墙是用 AMB3R 抬到三维的，用它自己的位姿投回去才对得上（实测 6–16 px，GT 位姿 24–79 px）。

## 哪些值是假定的

| 值 | 取值 | 依据 |
|---|---|---|
| 层高 `storey.height_m` | {storey:.2f} m | **假定**。没有一帧拍到墙和天花板的交线；看到的墙点最高 {top:.2f} m，只能说明墙不低于它 |
| 外墙厚度 | {te:.2f} m | **假定**。外墙背后从没被看到 |
| 内墙厚度 | {ti:.2f} m | **假定**。`evidence.thickness_hint` 里有间接线索（只作参考） |
| 推断段 | `segments[].kind = inferred` | 把观测到的墙延长、补齐成闭合轮廓的部分 |

## 各面墙

| 墙 | 类型 | 长度 | 观测 / 推断 | 关键帧（视角） | 关键帧拍到：观测段 / 全墙 |
|---|---|---|---|---|---|
{rows}

## 已知问题

{issues}
"""


def known_issues_zh(doc: dict) -> list[str]:
    w = {x["id"]: x for x in doc["walls"]}
    share = lambda i: 100 * w[i]["evidence"]["inferred_m"] / w[i]["length_m"]
    return [
        "整个房间每边比 ScanNet GT 小 3–16 cm（抬升的尺度误差，这一步没有修正；GT 只用于评估）",
        f"层高 {doc['storey']['height_m']:.2f} m 和墙厚都是假定值，不是测出来的",
        f"ext_W 有 {share('ext_W'):.0f}% 是推断补的，ext_N 有 {share('ext_N'):.0f}%：那几段没有任何墙 mask",
        "关键帧的相机位姿是 AMB3R 的估计；`reprojection_px` 表示每一帧的位姿和像素对得多准",
        "关键帧没有按运动模糊筛过：一帧可能整体清晰、局部糊（ext_N third，第 4582 帧，墙旁边的柜子是糊的）。"
        "`clarity` 是整帧拉普拉斯方差除以 50 帧中位数，和物体交接用的是同一个量，抓不到局部模糊",
        "ext_E 的关键帧只拍到观测段的一半左右：北段 4.2–6.1 m（厨房那一角）在点云里有零星墙点，"
        "但没有一帧在那里看到 ≥ 0.25 m² 的墙面，墙被橱柜切成了小块",
    ]


# ---------------------------------------------------------------- main
def build(run: str, output: Path, scene: Path = SCENE, runs: Path = MASK_RUNS, rectified: bool = False) -> dict:
    geom = OUT_ROOT / run
    closed = json.loads((geom / "closed.json").read_text())
    planes = json.loads((geom / "walls.json").read_text())
    meta = read_scene_meta(scene)
    camera = schema.camera(meta["colorWidth"], meta["colorHeight"], meta["fx_color"], meta["fy_color"],
                           meta["mx_color"], meta["my_color"],
                           source=f"{scene / (scene.name + '.txt')} (fx_color, fy_color, mx_color, my_color, "
                                  "colorWidth, colorHeight); the same intrinsics as the object handoff")
    width, height = int(camera["width"]), int(camera["height"])
    frames = Frames(planes)
    frames.K = np.asarray(camera["K"], dtype=np.float64)
    slot = frame_slots()
    H = float(closed["storey"]["height_m"])

    walls = build_walls(closed, planes)
    print(f"{len(walls)} walls; scoring key frames over {len(list((runs / run).glob('frame_*')))} segmented frames", flush=True)
    cands = frame_candidates(walls, frames, runs / run, slot, width, height, H)

    if output.exists():
        shutil.rmtree(output)
    (output / "keyframes").mkdir(parents=True)
    clarity = {int(p.name.split("_")[1]): frame_clarity(FRAMES / f"frame_{int(p.name.split('_')[1]):06d}.jpg")
               for p in (runs / run).glob("frame_*")}
    clarity_median = float(np.median(list(clarity.values())))
    cam_xy = {}
    for w in walls:
        chosen, cols = choose(cands[w["id"]])
        obs = [s for s in w["segments"] if s["kind"] == "observed"]
        centres = (np.arange(int(np.ceil(w["length_m"] / COVER_BIN)) + 1) + 0.5) * COVER_BIN
        centres = centres[centres < w["length_m"]]
        seen = np.zeros(len(centres), bool) if cols is None else cols[:len(centres)]
        in_obs = np.zeros(len(centres), bool)
        for sg in obs:
            in_obs |= (centres >= sg["from_m"]) & (centres < sg["to_m"])
        w["evidence"]["key_frame_candidates"] = len(cands[w["id"]])
        w["evidence"]["key_frames_show"] = {
            "observed_share": round(float(seen[in_obs].mean()) if in_obs.any() else 0.0, 3),
            "wall_share": round(float(seen.mean()), 3),
            "note": f"share of {COVER_BIN * 100:.0f} cm stretches along the wall that at least one key frame shows"}
        (output / "keyframes" / w["id"]).mkdir()
        for role, c in zip(schema.KEYFRAME_ROLES, chosen):
            stem = f"keyframes/{w['id']}/{c['seq']:04d}"
            shutil.copyfile(FRAMES / f"frame_{c['fid']:06d}.jpg", output / f"{stem}_rgb.jpg")
            Image.fromarray((c["mask"] * 255).astype(np.uint8), "L").save(output / f"{stem}_mask.png")
            rgb = Image.open(FRAMES / f"frame_{c['fid']:06d}.jpg").convert("RGB")
            # 4:4:4 chroma: with the default 4:2:0 the yellow ring bled into the wall pixels beside it
            draw_boxed(rgb, c["mask"], f"{w['id']} {role}").save(output / f"{stem}_boxed.jpg", quality=95, subsampling=0)
            ys, xs = np.nonzero(c["mask"])
            rec = {"role": role, "seq": c["seq"], "source_frame_id": c["fid"],
                   "rgb": f"{stem}_rgb.jpg", "mask": f"{stem}_mask.png", "boxed": f"{stem}_boxed.jpg",
                   "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                   "bbox_convention": schema.BBOX_CONVENTION, "visible_pixels": int(c["mask"].sum()),
                   "visible_area_m2": round(c["area"], 3),
                   "covers_m": [round(c["cover"][0], 3), round(c["cover"][1], 3)],
                   "view_angle_deg": round(c["view"], 1), "distance_m": round(c["dist"], 3),
                   "plane_offset_m": {k: round(v, 4) for k, v in c["offsets"].items()},
                   "reprojection_px": round(c["reproj"], 1),
                   "clarity": round(clarity[c["fid"]] / clarity_median, 2),
                   "camera_to_world": [[round(float(x), 8) for x in row] for row in c["pose"]]}
            if rectified and c["view"] <= RECT_MAX_DEG:
                image, along, hh = rectify(w, c, frames, rgb, H)
                image.save(output / f"{stem}_rectified.png")
                rec["rectified"] = {"image": f"{stem}_rectified.png", "px_per_m": RECT_PX_PER_M,
                                    "along_m": along, "height_m": hh,
                                    "note": "x = metres along the wall from start, y = height (top row = height_m); "
                                            "alpha = this wall's mask"}
            w["key_frames"].append(rec)
            cam_xy[(w["id"], c["fid"])] = c["cam"][:2]
        if obs and not chosen:
            print(f"  {w['id']}: observed but no key frame passed the thresholds")

    rect = closed["rectangle"]
    ext = {w["id"]: w for w in walls if w["role"] == "exterior"}
    inner = [ext[k]["start_floor"] for k in ("ext_S", "ext_E", "ext_N", "ext_W")]
    outer = [ext[k]["footprint_floor"][3] for k in ("ext_S", "ext_E", "ext_N", "ext_W")]
    up = frames.floor_to_scan[:3, 2]; origin = frames.floor_to_scan[:3, 3]
    doc = {
        "schema_version": ws.SCHEMA_VERSION, "scene_id": scene.name,
        "producer": {"name": "harness_structure", "run": run,
                     "note": "walls lifted from 2D wall masks (point-prompt segmentation) with AMB3R depth, "
                             "planes found per mask and cut where the scene's rays see through, then closed "
                             "with a rectangle; GT was not read",
                     "sources": {"closed_walls": str(geom / "closed.json"),
                                 "wall_planes": str(geom / "walls.json"),
                                 "masks": str(runs / run), "rgb": str(FRAMES), "point_maps": str(DEFAULT_NPZ),
                                 "sim3": str(DEFAULT_SIM3)}},
        "coordinate_frame": "scannet_scan", "units": "meters", "up_axis": "Z",
        "floor_frame_to_scan": [[round(float(x), 9) for x in row] for row in frames.floor_to_scan],
        "floor_frame_note": "p_scan = floor_frame_to_scan @ [x, y, z, 1]; floor frame: z = 0 on the floor plane, "
                            "+z up, x/y as the ScanNet axis-aligned frame turned by the floor tilt",
        "camera": camera,
        "storey": {"floor_z_floor": 0.0, "floor_plane_scan": {"point": [round(float(x), 5) for x in origin],
                                                             "normal": [round(float(x), 6) for x in up]},
                   "height_m": H, "kind": closed["storey"]["kind"],
                   "evidence": {"observed_wall_top_m": round(closed["storey"]["observed_wall_top_m"], 3),
                                "why_not_measured": closed["storey"]["why"]}},
        "thickness_defaults": {"exterior_m": closed["thickness"]["exterior_m"],
                               "interior_m": closed["thickness"]["interior_m"], "kind": "assumed",
                               "applied": "behind the room-side face, along normal_to_back; the face stays where it was seen"},
        "outline": {"shape": "rectangle", "axis_deg": round(closed["axis_deg"], 3),
                    "inner_corners_floor": inner, "outer_corners_floor": outer,
                    "order": "counter-clockwise from the south-west corner: SW, SE, NE, NW",
                    "inner_size_m": [round(rect["width_m"], 3), round(rect["depth_m"], 3)],
                    "checks": {"seen_space_outside_share": round(rect["seen_outside_share"], 4),
                               "inside_seen_share": round(rect["inside_seen_share"], 3),
                               "largest_unseen_inside_m2": round(rect["largest_unseen_inside_m2"], 2)}},
        "walls": [{k: v for k, v in w.items() if not k.startswith("_")} for w in walls],
        "not_included": list(ws.NOT_INCLUDED),
        "known_issues": [
            "the room comes out 3-16 cm smaller per side than the ScanNet GT (lifting scale error, "
            "not corrected here; measured against GT for evaluation only)",
            f"storey height {H:.2f} m and wall thickness are assumed, not measured",
            "ext_W is %.0f%% inferred and ext_N %.0f%% inferred: no wall mask covers those stretches"
            % (100 * ext["ext_W"]["evidence"]["inferred_m"] / ext["ext_W"]["length_m"],
               100 * ext["ext_N"]["evidence"]["inferred_m"] / ext["ext_N"]["length_m"]),
            "key-frame poses are AMB3R estimates; reprojection_px says how well each one fits",
            "key frames are not screened for motion blur: a frame can be sharp overall and smeared in part "
            "(ext_N third, frame 4582, smears the cabinets beside the wall); `clarity` is the whole-frame "
            "Laplacian variance over the median of the segmented frames, the same measure the object handoff uses",
        ],
    }
    (output / "walls.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    preview(doc, cam_xy, output / "preview.png")
    rows = []
    for w in doc["walls"]:
        obs = w["evidence"]["observed_m"]
        kfs = "、".join(f"{k['source_frame_id']}（{k['view_angle_deg']:.0f}°）" for k in w["key_frames"]) or "—"
        rows.append(f"| {w['id']} | {'外墙' if w['role'] == 'exterior' else '内墙'} {int(round(w['thickness_m'] * 100))} cm "
                    f"| {w['length_m']:.2f} m | {obs:.2f} / {w['evidence']['inferred_m']:.2f} m | {kfs} "
                    f"| {w['evidence']['key_frames_show']['observed_share']:.0%} / {w['evidence']['key_frames_show']['wall_share']:.0%} |")
    (output / "README.md").write_text(README.format(
        scene=scene.name, run=run, out=output, w=width, h=height, storey=H,
        top=closed["storey"]["observed_wall_top_m"], te=closed["thickness"]["exterior_m"],
        ti=closed["thickness"]["interior_m"], rows="\n".join(rows),
        issues="\n".join(f"- {x}" for x in known_issues_zh(doc))))
    print(f"wrote {output}")
    for w in doc["walls"]:
        print(f"  {w['id']:6} {w['length_m']:.2f} m  key frames "
              + ", ".join(f"{k['source_frame_id']}({k['role'][0]}, {k['view_angle_deg']:.0f}°, "
                          f"{k['covers_m'][0]:.1f}-{k['covers_m'][1]:.1f} m, reproj {k['reprojection_px']:.0f}px)"
                          for k in w["key_frames"]))
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="struct_v4_win5")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--rectified", action="store_true", help="also write square-on views of each wall (<=60 deg)")
    a = ap.parse_args()
    build(a.run, a.output.resolve(), a.scene, rectified=a.rectified)


if __name__ == "__main__":
    main()
