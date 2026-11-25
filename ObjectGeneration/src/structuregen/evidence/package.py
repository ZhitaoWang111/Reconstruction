"""Stage 2: what the appearance agent is given before it starts. No model, no money.

    PYTHONPATH=src .venv-sam3d/bin/python -m structuregen.evidence.package [run dir] [--first <segment>]

Only facts from the handoff and plain geometry of the white model; nothing inferred (no rooms,
no colours, no trim). Whether a wall has skirting or a door casing, and what colour it is, is
the agent's call from the photos.

    evidence/task_card.md      the text: every segment with its number, Blender name, ends,
                               length, thickness, height, room-side normal, openings, and the
                               handoff's key frames for its wall
    evidence/task_card.json    the same as data
    evidence/plan.png          top view, every segment in its own colour with its number
    evidence/first_wall.json   which segment the agent starts with, and why (the ranking)
    evidence/first/            that segment's key frames, copied from the handoff verbatim:
                               <seq>_rgb.jpg (original) and <seq>_boxed.jpg (wall outlined)

The agent then works wall by wall; the next wall's frames are handed over the same way.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .. import config
from ..common import elevation as el, handoff as handoff_io

CELL_M = 0.02                    # wall-surface grid for the coverage measure
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ------------------------------------------------------------------------------- facts

def frames_for(wall: dict, seg: dict) -> list[dict]:
    """The handoff key frames of the segment's wall whose covers_m overlaps the segment."""
    out = []
    for k in wall["key_frames"]:
        lo, hi = k["covers_m"]
        overlap = max(0.0, min(hi, seg["a1"]) - max(lo, seg["a0"]))
        if overlap > 0:
            out.append({"frame": k["source_frame_id"], "seq": k["seq"], "role": k["role"],
                        "covers_m": k["covers_m"], "overlap_m": round(overlap, 3),
                        "view_angle_deg": k["view_angle_deg"], "reprojection_px": k.get("reprojection_px"),
                        "mask_agreement": k.get("mask_agreement"),
                        "visible_area_m2": k["visible_area_m2"], "visible_pixels": k["visible_pixels"],
                        "clarity": k.get("clarity"), "rgb": k["rgb"], "boxed": k["boxed"], "mask": k["mask"]})
    return out


def mask_coverage(handoff: Path, camera: dict, wall: dict, seg: dict, frames: list[dict]) -> dict:
    """Square metres of this segment's room-side face that the handoff masks cover.

    The face is sampled every CELL_M; a sample counts for a frame when its pixel is inside that
    frame's mask and the camera is on the room side. Pure projection of handoff data.
    """
    a = np.arange(seg["a0"] + CELL_M / 2, seg["a1"], CELL_M)
    z = np.arange(CELL_M / 2, seg["height_m"], CELL_M)
    A, Z = np.meshgrid(a, z)
    pts = el.wall_point(wall, A, Z)
    room_side = -np.array(wall["normal_to_back"][:2], float)
    union = np.zeros(A.shape, bool)
    per_frame = {}
    for f in frames:
        c2w = np.array(f["_c2w"], float)
        mask = cv2.imread(str(handoff / f["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        u, v, d = el.project(camera, c2w, pts)
        facing = ((c2w[:2, 3] - pts[..., :2]) @ room_side) > 0
        ok = (d > 0.05) & (u >= 0) & (u < camera["width"]) & (v >= 0) & (v < camera["height"]) & facing
        hit = np.zeros(A.shape, bool)
        hit[ok] = mask[v[ok].astype(int), u[ok].astype(int)]
        per_frame[f["frame"]] = round(float(hit.sum() * CELL_M ** 2), 3)
        union |= hit
    face = (seg["a1"] - seg["a0"]) * seg["height_m"]
    return {"seen_m2": round(float(union.sum() * CELL_M ** 2), 3), "face_m2": round(face, 3),
            "seen_share": round(float(union.mean()), 3), "per_frame_m2": per_frame}


def rank_first(segments: list[dict]) -> list[dict]:
    """Order segments for 'which wall first': exterior before interior, then the most wall
    seen in the masks, then the best single view (straightest, best pose). Inputs are handoff
    fields and their projection only."""
    def best_view(s):
        views = [f for f in s["frames"] if s["coverage"]["per_frame_m2"].get(f["frame"], 0) > 0]
        return min(views, key=lambda f: (f["view_angle_deg"], f["reprojection_px"] or 0)) if views else None

    rows = []
    for s in segments:
        b = best_view(s)
        rows.append({"segment": s["id"], "number": s["number"], "role": s["role"],
                     "frames": sum(1 for v in s["coverage"]["per_frame_m2"].values() if v > 0),
                     "seen_m2": s["coverage"]["seen_m2"], "seen_share": s["coverage"]["seen_share"],
                     "best_frame": b["frame"] if b else None,
                     "best_view_angle_deg": b["view_angle_deg"] if b else None,
                     "best_reprojection_px": b["reprojection_px"] if b else None,
                     "best_frame_seen_m2": s["coverage"]["per_frame_m2"].get(b["frame"]) if b else 0})
    rows.sort(key=lambda r: (r["role"] != "exterior", -r["seen_m2"], r["best_view_angle_deg"] or 99))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# ------------------------------------------------------------------------------- floors

FLOOR_CELL_M = 0.05
FLOOR_FRAMES = 3


def _blocked_by_walls(cam_xyz, pts_xy, segments) -> np.ndarray:
    """For each floor point: does the sight line from the camera cross a wall body first?

    Plan-view test against every segment's footprint quad: the line from the camera (above the
    floor) down to the point is blocked where it passes over a footprint, unless it passes there
    through an opening and below the opening's top. Walls are full height and every camera is
    below the storey height, so the plan view is enough. Geometry of the white model only.
    """
    c = np.asarray(cam_xyz[:2], float)
    h = float(cam_xyz[2])
    blocked = np.zeros(len(pts_xy), bool)
    d = pts_xy - c
    for seg in segments:
        f0, f1 = np.array(seg["front"][0]), np.array(seg["front"][1])
        quad = [f0, f1, np.array(seg["back"][1]), np.array(seg["back"][0])]
        u = (f1 - f0) / np.linalg.norm(f1 - f0)
        # parameter t along camera->point where it enters / leaves the convex quad (Cyrus-Beck)
        t0, t1 = np.zeros(len(d)), np.ones(len(d))
        inside_all = np.ones(len(d), bool)
        for k in range(4):
            a_, b_ = quad[k], quad[(k + 1) % 4]
            e = b_ - a_
            n = np.array([e[1], -e[0]])
            centre = sum(quad) / 4
            if n @ (centre - a_) > 0:
                n = -n                                        # outward normal
            num = n @ (a_ - c)
            den = d @ n
            par = np.abs(den) < 1e-12
            inside_all &= ~(par & (num < 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                t = num / den
            t0 = np.where(~par & (den < 0), np.maximum(t0, t), t0)
            t1 = np.where(~par & (den > 0), np.minimum(t1, t), t1)
        hit = inside_all & (t0 < t1 - 1e-9) & (t0 < 1 - 1e-6)
        if not hit.any():
            continue
        tm = (t0 + np.minimum(t1, 1)) / 2                     # middle of the crossing
        along = ((c + d * tm[:, None]) - f0) @ u + seg["a0"]
        z = h * (1 - tm)                                       # height of the sight line there
        through = np.zeros(len(d), bool)
        for o in seg["openings"]:
            through |= (along > o["from_m"]) & (along < o["to_m"]) & (z < o["z_floor"][1]) & (z > o["z_floor"][0])
        blocked |= hit & ~through
    return blocked


def floor_frames(handoff: Path, doc: dict, floor: dict, segments: list) -> list[dict]:
    """Key frames that show most of this floor, from the white model's geometry.

    The floor's top polygons are sampled every FLOOR_CELL_M and projected into every key frame of
    the handoff; a sample counts when it lands in the image in front of the camera and no wall of
    the white model stands between the camera and it. Furniture is not known here, so this is
    floor in view, not floor visible. Frames are picked greedily, each adding the most samples
    not yet seen.
    """
    from matplotlib.path import Path as MPath
    pts = []
    for poly in floor["top_polygons"]:
        poly = np.array(poly)
        lo, hi = poly.min(0), poly.max(0)
        xs, ys = np.meshgrid(np.arange(lo[0] + FLOOR_CELL_M / 2, hi[0], FLOOR_CELL_M),
                             np.arange(lo[1] + FLOOR_CELL_M / 2, hi[1], FLOOR_CELL_M))
        cand = np.c_[xs.ravel(), ys.ravel()]
        pts.append(cand[MPath(poly).contains_points(cand)])
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    if not len(pts):
        return []
    world = np.c_[pts, np.zeros(len(pts)), np.ones(len(pts))]
    cam = doc["camera"]
    seen = {}
    for w in doc["walls"]:
        for k in w["key_frames"]:
            fid = k["source_frame_id"]
            if fid in seen:
                continue
            c2w = np.array(k["camera_to_world"], float)
            u, v, dd = el.project(cam, c2w, world)
            ok = (dd > 0.1) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
            ok &= ~_blocked_by_walls(c2w[:3, 3], pts, segments)
            seen[fid] = {"keep": ok, "rgb": k["rgb"]}
    picked, union = [], np.zeros(len(pts), bool)
    for _ in range(FLOOR_FRAMES):
        best = max(seen, key=lambda fid: (seen[fid]["keep"] & ~union).sum())
        gain = int((seen[best]["keep"] & ~union).sum())
        if gain < 0.02 * len(pts):
            break
        union |= seen[best]["keep"]
        picked.append({"frame": best, "rgb": seen[best]["rgb"],
                       "in_view_m2": round(float(seen[best]["keep"].sum()) * FLOOR_CELL_M ** 2, 2),
                       "new_m2": round(gain * FLOOR_CELL_M ** 2, 2)})
    for i, f in enumerate(picked):
        f["role"] = ("primary", "secondary", "third")[i]
        f["in_view_share"] = round(float(union.mean()), 3)
    return picked


def floor_overlay(handoff: Path, doc: dict, floor: dict, frame: dict, path: Path, segments: list) -> None:
    """The key frame with this floor's outline drawn where no wall hides it, like the walls' _boxed.jpg."""
    k = next(k for w in doc["walls"] for k in w["key_frames"] if k["source_frame_id"] == frame["frame"])
    img = cv2.imread(str(handoff / k["rgb"]))
    c2w = np.array(k["camera_to_world"], float)
    for poly in floor["top_polygons"]:
        poly = np.array(poly)
        dense = []
        for a, b in zip(poly, np.roll(poly, -1, 0)):
            n = max(2, int(np.linalg.norm(b - a) / 0.02))
            dense.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        dense = np.concatenate(dense)
        u, v, d = el.project(doc["camera"], c2w, np.c_[dense, np.zeros(len(dense)), np.ones(len(dense))])
        # an outline point on the room's edge sits against a wall: test a point 3 cm inside it
        inward = dense + (np.array(poly).mean(0) - dense) / np.maximum(np.linalg.norm(np.array(poly).mean(0) - dense, axis=1), 1e-9)[:, None] * 0.03
        ok = (d > 0.1) & ~_blocked_by_walls(c2w[:3, 3], inward, segments)
        for i in range(len(dense)):
            j = (i + 1) % len(dense)
            if ok[i] and ok[j]:
                cv2.line(img, (int(u[i]), int(v[i])), (int(u[j]), int(v[j])), (0, 220, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ------------------------------------------------------------------------------- plan view

def plan_view(doc: dict, segments: list[dict], path: Path, first: str | None, floors: list | None = None) -> None:
    """Top view in the handoff's frame: each segment filled in its colour with its number on it, its name
    outside the room, an arrow along its room-side normal, openings dashed, axes and a 1 m bar."""
    pts = np.array([p for s in segments for p in s["front"] + s["back"]], float)
    lo, hi = pts.min(0) - 0.9, pts.max(0) + 0.9
    S = 110.0                                            # pixels per metre
    W, H = int((hi[0] - lo[0]) * S), int((hi[1] - lo[1]) * S)
    xy = lambda p: ((p[0] - lo[0]) * S, (hi[1] - p[1]) * S)
    img = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(img)
    num_font, name_font = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT_REGULAR, 16)

    for x in range(int(math.floor(lo[0])), int(math.ceil(hi[0])) + 1):      # 1 m grid
        dr.line([xy((x, lo[1])), xy((x, hi[1]))], fill=(234, 234, 234))
    for y in range(int(math.floor(lo[1])), int(math.ceil(hi[1])) + 1):
        dr.line([xy((lo[0], y)), xy((hi[0], y))], fill=(234, 234, 234))

    def arrow(p0, p1, colour, width=2):
        (x0, y0), (x1, y1) = xy(p0), xy(p1)
        dr.line([(x0, y0), (x1, y1)], fill=colour, width=width)
        ang = math.atan2(y1 - y0, x1 - x0)
        for d in (2.6, -2.6):
            dr.line([(x1, y1), (x1 + 11 * math.cos(ang + d), y1 + 11 * math.sin(ang + d))], fill=colour, width=width)

    for f in floors or []:                               # floors under the walls, light, named
        for poly in f["top_polygons"]:
            dr.polygon([xy(p) for p in poly], fill=(236, 232, 222) if f["kind"] == "room" else (214, 206, 190),
                       outline=(170, 160, 140))
        if f["kind"] == "room":
            big = max(f["top_polygons"], key=lambda q: len(q))
            c = np.array(big).mean(0)
            dr.text(xy(c), f["id"], font=num_font, fill=(150, 140, 120), anchor="mm")
    for s in segments:
        quad = [xy(p) for p in (s["front"][0], s["front"][1], s["back"][1], s["back"][0])]
        dr.polygon(quad, fill=tuple(s["colour"]), outline=(40, 40, 40))
        if s["id"] == first:
            dr.polygon(quad, outline=(0, 0, 0), width=5)
    for s in segments:                                   # openings: gap in the wall, dashed outline
        w = s["_wall"]
        for o in s["openings"]:
            a = el.wall_point(w, np.array([o["from_m"], o["to_m"]]), np.zeros(2), w["thickness_m"] / 2)
            dr.line([xy(a[0]), xy(a[1])], fill=(250, 250, 250), width=int(w["thickness_m"] * S) + 3)
            n = int(np.linalg.norm(a[1][:2] - a[0][:2]) / 0.06)
            for k in range(0, n, 2):
                q0 = a[0][:2] + (a[1][:2] - a[0][:2]) * k / n
                q1 = a[0][:2] + (a[1][:2] - a[0][:2]) * (k + 1) / n
                dr.line([xy(q0), xy(q1)], fill=(60, 60, 60), width=2)
    for s in segments:                                   # normals, numbers, names on top
        f0, f1 = np.array(s["front"][0]), np.array(s["front"][1])
        b0, b1 = np.array(s["back"][0]), np.array(s["back"][1])
        n = np.array(s["normal_room"][:2])
        mid_front = (f0 + f1) / 2
        arrow(mid_front, mid_front + n * min(0.45, 0.35 * s["length_m"] + 0.1), (30, 30, 30))
        centre = (f0 + f1 + b0 + b1) / 4
        x, y = xy(centre)
        r = 15
        dr.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=(20, 20, 20), width=2)
        dr.text((x, y), str(s["number"]), font=num_font, fill=(20, 20, 20), anchor="mm")
        # names: exterior walls outside the room, interior walls on their room side past the arrow
        along = (f1 - f0) / (np.linalg.norm(f1 - f0) or 1)
        offset = -n * 0.45 if s["role"] == "exterior" else n * 0.75 + along * 0.0
        dr.text(xy(centre + offset), s["id"], font=name_font, fill=(50, 50, 50), anchor="mm")

    ox, oy = 30, H - 30                                  # axes and scale bar
    dr.line([(ox, oy), (ox + 50, oy)], fill=(220, 50, 50), width=3)
    dr.text((ox + 56, oy), "+X", font=name_font, fill=(220, 50, 50), anchor="lm")
    dr.line([(ox, oy), (ox, oy - 50)], fill=(40, 160, 40), width=3)
    dr.text((ox, oy - 58), "+Y", font=name_font, fill=(40, 160, 40), anchor="mb")
    dr.line([(W - 30 - S, H - 30), (W - 30, H - 30)], fill=(40, 40, 40), width=3)
    dr.text((W - 30 - S / 2, H - 36), "1 m", font=name_font, fill=(40, 40, 40), anchor="mb")
    img.save(path)


# ------------------------------------------------------------------------------- card

def card_text(card: dict) -> str:
    f2 = lambda p: f"({p[0]:.3f}, {p[1]:.3f})"
    L = [f"# 任务卡 · {card['scene']} 墙面外观", "",
         f"坐标系 {card['scene_facts']['frame']}：米，+Z 向上，地面 z = 0。墙段编号见俯视图 plan.png（每段一种颜色）。",
         "每段在 Blender 里是一个独立物体，名字 = 段名；它的每个面是一个材质槽，名字为 <段名>_<面>，"
         "面有 front（房间一侧）、back、top、bottom、end_start、end_end、reveal（洞口侧壁）。", "",
         "## 墙段", "",
         "| # | 段名 | 所属墙 | 类型 | front 起点 → 终点 (x, y) | 长 m | 厚 m | 高 m | front 法向 (x, y, z) | 洞口 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in card["segments"]:
        op = "；".join(f"沿墙 {o['from_m']:.3f}–{o['to_m']:.3f} m，高 {o['z_m'][0]}–{o['z_m'][1]} m"
                      for o in s["openings"]) or "—"
        n = s["normal_room"]
        L.append(f"| {s['number']} | {s['id']} | {s['wall']} | {'外墙' if s['role'] == 'exterior' else '内墙'} | "
                 f"{f2(s['front'][0])} → {f2(s['front'][1])} | {s['length_m']:.3f} | {s['thickness_m']} | "
                 f"{s['height_m']} | ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.0f}) | {op} |")
    if card.get("floors"):
        L += ["", "## 地板", "",
              "地板在 Blender 里也是独立物体，名字 = 地板名，面有 top（朝上，房间里看到的那一面）、side、bottom。",
              "", "| 地板 | 类型 | 面积 m² | 关键帧（露出的地面 m²） |", "|---|---|---|---|"]
        for f in card["floors"]:
            fr = "、".join(f"{x['frame']}（{x['in_view_m2']}）" for x in f["frames"]) or "—"
            kind = "房间" if f["kind"] == "room" else f"门槛（{f.get('opening_of')} 的洞口下）"
            L.append(f"| {f['id']} | {kind} | {f['area_m2']} | {fr} |")
        L.append("")
        L.append("关键帧按「在视野内、没被墙挡住的地面面积」挑选（按白模几何算）；家具会挡住一部分，图里黄线圈出这块地板露出的边界。")
    L += ["", "front 法向指向墙的房间一侧。沿墙距离从所属墙的起点量起（段 _2 的沿墙范围接着 _1）。", "",
          "## 每段的关键帧（交接原样）", "",
          "| # | 段名 | 帧 | role | 拍到沿墙 m | 视角° | 重投影 px |", "|---|---|---|---|---|---|---|"]
    for s in card["segments"]:
        for f in s["frames"]:
            L.append(f"| {s['number']} | {s['id']} | {f['frame']} | {f['role']} | {f['covers_m'][0]}–{f['covers_m'][1]} | "
                     f"{f['view_angle_deg']} | {f['reprojection_px'] if f['reprojection_px'] is not None else '—'} |")
        if not s["frames"]:
            L.append(f"| {s['number']} | {s['id']} | — | — | — | — | — |")
    L += ["", "视角是视线与墙面法向的夹角（0° 为正对）；重投影是交接给出的位姿误差，越小越可信（用 GT 位姿的交接没有这一项，记为 —）。", "",
          "## 交接里的假定与缺失", "", "| 项 | 值 | 性质 | 备注 |", "|---|---|---|---|"]
    for a in card["assumed"]:
        L.append(f"| {a['item']} | {a['value']} | {a['kind']} | {a['note']} |")
    L += ["", "交接不含：" + "、".join(card["not_included"]) + "。openings 只有测到的洞口；关着的门、玻璃窗处仍是实墙。"]
    L += ["", "## 做法", "",
          "先逐块做地板（每个房间一块，一种铺面：纯色，或瓷砖 / 木地板这类重复纹理），再逐段做墙。"
          "每段先看它的原图和带墙框的图，再决定这一段墙的外观：颜色、材质，以及墙上是否有"
          "踢脚线、腰线、洞口的门套，在哪、多大。每次改动后会自动收到检查结果和照片对比图。做完一段再进入下一段。",
          "门扇和窗（含窗套）由实例物体生成负责，不在这一步做：照片里有门窗的地方，墙保持实墙，不开洞、不画框。"
          "只有上表「洞口」一栏里的洞是开着的。",
          f"第一面：#{card['first']['number']} {card['first']['segment']}，图在 first/ 下。"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?", default=None)
    parser.add_argument("--first", help="Segment id to start with; default the top of the ranking")
    args = parser.parse_args(argv)
    run = (args.run or (config.OUTPUT_ROOT / config.SCENE / "structuregen/latest")).resolve()
    plan = json.loads((run / "plan.json").read_text())
    handoff = Path(plan["handoff_dir"])
    doc = handoff_io.load(handoff)
    walls = {w["id"]: w for w in doc["walls"]}
    camera = doc["camera"]
    c2w_of = {k["source_frame_id"]: k["camera_to_world"] for w in doc["walls"] for k in w["key_frames"]}

    segments = []
    for i, seg in enumerate(plan["segments"]):
        wall = walls[seg["wall"]]
        nb = np.array(wall["normal_to_back"][:2], float)
        frames = frames_for(wall, seg)
        for f in frames:
            f["_c2w"] = c2w_of[f["frame"]]
        s = {"number": i + 1, "id": seg["id"], "wall": seg["wall"], "role": seg["role"],
             "front": seg["front"], "back": seg["back"],
             "along_m": [round(seg["a0"], 3), round(seg["a1"], 3)], "length_m": round(seg["length_m"], 3),
             "thickness_m": seg["thickness_m"], "height_m": seg["height_m"],
             "normal_room": [round(float(-nb[0]), 4), round(float(-nb[1]), 4), 0.0],
             "openings": [{"from_m": o["from_m"], "to_m": o["to_m"], "z_m": o["z_floor"]} for o in seg["openings"]],
             "colour": list(config.SEGMENT_COLOURS[i % len(config.SEGMENT_COLOURS)]),
             "frames": frames, "_wall": wall}
        s["coverage"] = mask_coverage(handoff, camera, wall, seg, frames)
        segments.append(s)

    ranking = rank_first(segments)
    first_id = args.first or ranking[0]["segment"]
    first = next(s for s in segments if s["id"] == first_id)

    out = run / "evidence"
    if out.exists():
        shutil.rmtree(out)
    (out / "first").mkdir(parents=True)
    floors = []
    (out / "floors").mkdir()
    for f in plan.get("floors", []):
        fr = floor_frames(handoff, doc, f, plan["segments"])
        for x in fr:
            x["boxed"] = f"floors/{f['id']}_{x['frame']:04d}_boxed.jpg"
            floor_overlay(handoff, doc, f, x, out / x["boxed"], plan["segments"])
        floors.append({"id": f["id"], "kind": f["kind"], "area_m2": round(f["area_m2"], 3),
                       "top_polygons": f["top_polygons"], "frames": fr,
                       **({"opening_of": f["sg_opening_of"]} if "sg_opening_of" in f else {})})
    for f in floors:                      # a threshold joins the rooms on either side of its opening
        if f["kind"] != "threshold":
            continue
        from matplotlib.path import Path as MPath
        seg = next(x for x in plan["segments"] if x["id"] == f["opening_of"])
        f0, f1, b0 = (np.array(v, float) for v in (seg["front"][0], seg["front"][1], seg["back"][0]))
        u = (f1 - f0) / np.linalg.norm(f1 - f0)
        n = (b0 - f0) - u * ((b0 - f0) @ u)
        n /= np.linalg.norm(n)
        c = np.array(f["top_polygons"][0]).mean(0)
        f["between"] = []
        for side in (-1, 1):
            q = c + n * side * (seg["thickness_m"] / 2 + 0.05)
            room = next((r["id"] for r in floors if r["kind"] == "room"
                         and any(MPath(np.array(poly)).contains_point(q) for poly in r["top_polygons"])), None)
            f["between"].append(room)
    plan_view(doc, segments, out / "plan.png", first_id, floors)

    copied = []
    for f in sorted(first["frames"], key=lambda f: -first["coverage"]["per_frame_m2"].get(f["frame"], 0)):
        for kind in ("rgb", "boxed"):
            src = handoff / f[kind]
            dst = out / "first" / f"{f['frame']:04d}_{kind}.jpg"
            shutil.copyfile(src, dst)
            copied.append(str(dst.relative_to(out)))

    clean = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}
    for s in segments:
        for f in s["frames"]:
            f.pop("_c2w", None)
    storey, thick = doc["storey"], doc["thickness_defaults"]
    first_frames = sorted(first["frames"], key=lambda f: -first["coverage"]["per_frame_m2"].get(f["frame"], 0))
    card = {
        "schema": "structuregen/task_card/v2", "scene": doc["scene_id"],
        "scene_facts": {"frame": doc["coordinate_frame"], "units": doc["units"], "up_axis": doc["up_axis"],
                        "floor_z": storey["floor_z_floor"], "segments": len(segments),
                        "exterior": sum(s["role"] == "exterior" for s in segments),
                        "interior": sum(s["role"] == "interior" for s in segments),
                        "openings": sum(len(s["openings"]) for s in segments),
                        "key_frames": len({f["frame"] for s in segments for f in s["frames"]})},
        "blender": {"object": "<段名>", "material_slot": "<段名>_<面>",
                    "faces": ["front", "back", "top", "bottom", "end_start", "end_end", "reveal"],
                    "floor_faces": ["top", "side", "bottom"]},
        "segments": [clean(s) for s in segments],
        "floors": floors,
        "assumed": [
            {"item": "层高", "value": f"{storey['height_m']} m", "kind": storey["kind"],
             "note": f"看到的墙最高处 {storey['evidence']['observed_wall_top_m']} m"},
            {"item": "外墙厚", "value": f"{thick['exterior_m']} m", "kind": thick["kind"], "note": "加在 front 面背后"},
            {"item": "内墙厚", "value": f"{thick['interior_m']} m", "kind": thick["kind"], "note": "加在 front 面背后"},
        ],
        "not_included": doc["not_included"],
        "first": {"segment": first_id, "number": first["number"], "files": copied,
                  "frames": [{**f, "mask_area_m2": first["coverage"]["per_frame_m2"].get(f["frame"], 0)}
                             for f in first_frames],
                  "coverage": first["coverage"]},
    }
    (out / "task_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False))
    text = card_text(card)
    (out / "task_card.md").write_text(text)
    (out / "first_wall.json").write_text(json.dumps({"chosen": first_id, "ranking": ranking}, indent=1))

    print(f"EVIDENCE task card {len(text)} chars, plan.png, first wall #{first['number']} {first_id} "
          f"({len(copied) // 2} frames) -> {out}")
    for r in ranking:
        print(f"  {r['rank']}. #{r['number']} {r['segment']:8s} {r['role']:8s} frames {r['frames']} "
              f"seen {r['seen_m2']:.2f} m2 ({r['seen_share']:.0%})  best frame {r['best_frame']} "
              f"angle {r['best_view_angle_deg']} reproj {r['best_reprojection_px']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
