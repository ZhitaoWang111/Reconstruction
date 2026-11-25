"""Plan view: the walls with their ids, and each key frame's camera position and field of view.
No objects (user decision 2026-11-20: boxes and ids for objects make it unreadable)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


def plan_view(structure, cameras: dict, colours: dict | None = None, size=900, margin=60) -> Image.Image:
    """cameras: {seq: (c2w 4x4 OpenCV, horizontal fov in degrees)}."""
    pts = []
    for w in structure.walls.values():
        pts += [w.start, w.end]
    for f in structure.floors.values():
        for r in f.rings:
            pts += list(r)
    for c2w, _ in cameras.values():
        pts.append(np.asarray(c2w)[:2, 3])
    pts = np.array(pts, float)
    lo, hi = pts.min(0), pts.max(0)
    scale = (size - 2 * margin) / max(hi - lo)
    W = int((hi[0] - lo[0]) * scale + 2 * margin)
    H = int((hi[1] - lo[1]) * scale + 2 * margin)

    def px(p):                         # +Y up on the page
        return (margin + (p[0] - lo[0]) * scale, H - margin - (p[1] - lo[1]) * scale)

    im = Image.new('RGB', (W, H + 34), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f_small, f_big = ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT, 17)
    for fl in structure.floors.values():
        for r in fl.rings:
            d.polygon([px(p) for p in r], fill=(243, 240, 233) if fl.kind == 'room' else (230, 230, 230))
    for w in structure.walls.values():
        a, b = px(w.start), px(w.end)
        back_a = px(w.start - w.thickness * w.normal)
        back_b = px(w.end - w.thickness * w.normal)
        d.polygon([a, b, back_b, back_a], fill=(90, 96, 104))
        for o in w.openings:
            oa = px(w.start + o.from_m * w.direction)
            ob = px(w.start + o.to_m * w.direction)
            d.line([oa, ob], fill=(255, 255, 255), width=6)
        mid = (w.start + w.end) / 2 + 0.35 * w.normal
        tx, ty = px(mid)
        d.text((tx - 22, ty - 9), w.id, font=f_big, fill=(20, 60, 140))
    for seq, (c2w, fov) in sorted(cameras.items()):
        c2w = np.asarray(c2w)
        o = c2w[:2, 3]
        fwd = c2w[:2, 2]
        if np.linalg.norm(fwd) < 1e-6:
            continue
        fwd = fwd / np.linalg.norm(fwd)
        half = math.radians(fov / 2)
        rays = []
        for s in (-1, 1):
            c, si = math.cos(s * half), math.sin(s * half)
            rays.append(o + 1.2 * np.array([fwd[0] * c - fwd[1] * si, fwd[0] * si + fwd[1] * c]))
        col = (colours or {}).get(seq, (220, 80, 40))
        d.polygon([px(o), px(rays[0]), px(rays[1])], outline=col, width=3)
        cx, cy = px(o)
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=col)
        d.text((cx + 8, cy - 20), f'{seq}', font=f_small, fill=col)
    # north arrow / axes
    d.line([(20, H - 20), (70, H - 20)], fill=(200, 40, 40), width=3); d.text((74, H - 30), '+X', font=f_small, fill=(200, 40, 40))
    d.line([(20, H - 20), (20, H - 70)], fill=(40, 140, 60), width=3); d.text((12, H - 92), '+Y', font=f_small, fill=(40, 140, 60))
    d.text((10, H + 6), 'Plan view: walls (ids), openings (white gaps), key-frame cameras (dot = camera, wedge = view).',
           font=f_small, fill=(40, 40, 40))
    return im
