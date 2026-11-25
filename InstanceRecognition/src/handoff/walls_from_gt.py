"""Write the wall handoff (walls/v2) from the ScanNet annotation.

    python -m src.handoff.walls_from_gt                       # -> artifacts/handoff/scene0000_00/walls_gt

The counterpart of `walls_from_harness`: same contract, same `amb3r_floor` frame, so the
two directories overlay directly and can be compared wall for wall. Everything here is
measured on the annotated mesh; nothing is inferred from rays.

Why it exists: the harness producer can only report what a camera saw. Behind a curtain or
a closed door leaf there is no evidence, so it falls back on an assumed storey height,
assumed thicknesses, and it finds 1 of the room's 4 openings. The annotation has all of
them, plus the ceiling (so the storey height is measured) and both faces of every
partition (so interior thickness is measured). This directory is what the harness
directory is compared against, and what downstream can use as a reference build.

What is still not in it: the outside of an exterior wall was never annotated either, so its
thickness stays assumed; and a door leaf is one surface, so the opening it fills is taken
from the door frame that surrounds it, or from the leaf itself when no frame is annotated.

Steps
-----
1. Vertical wall vertices are grouped by the room's main axis (area-weighted, mod 90).
   Along each axis the offsets are binned at 10 cm and the peaks kept: a peak is a wall
   face. The face nearest the outside on each of the four axes is that side's exterior
   wall; every other peak is an interior wall.
2. Each wall face is refit on its own vertices and its room side fixed: an exterior wall
   faces into the room, an interior one faces the larger of the two floor areas beside it.
3. The four exterior faces make the closed rectangle, exactly as `close.py` does; interior
   walls are extended to the first line they meet, and the stretch they gain is `inferred`.
4. Openings come from the annotated instances: a door frame (with or without a leaf) or a
   window, expressed on the wall it hangs on. `source` is `gt_instance`, not `ray_test`.
5. Storey height is the annotated ceiling against the floor; thickness is measured between
   the two faces of a partition where both exist, assumed otherwise.
6. Key frames are the frames whose 2D instance mask shows the wall, picked by how much of
   it is visible and how distinct the views are - the same rules the object handoff uses.
"""
from __future__ import annotations

import argparse
import json
import zipfile
import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from src import config
from src.geometry.amb3r_mask_source import DEFAULT_SIM3
from src.geometry.floor_frame import Frame, scannet_scan_to_frame
from pagegen import build_structure_gt as gsg

from . import schema, walls_schema as ws
from .from_gt import draw_boxed, read_scene_meta, select_keyframes
from .walls_from_harness import build_walls, preview, README

SCENE = config.SCANNET_ROOT / "scene0000_00"
SOURCE = config.ARTIFACTS_ROOT / "sources/scene0000_00/scene_amb3r_v1"
FRAMES = config.ARTIFACTS_ROOT / "frames/scene0000_00"
GT_INPUT = config.WORKSPACE / "Video2PointCloud/data/scannet/scene0000_00_uniform500/input"
DEFAULT_OUTPUT = config.ARTIFACTS_ROOT / "handoff/scene0000_00/walls_gt"   # scannet_scan, like gt/

BIN = 0.10             # m; offset bins along an axis
PEAK_POINTS = 150      # vertices in a 10 cm bin before it can be a wall face (also the diagonal test)
GAP = 0.50             # m; a face's vertices may break along the wall by this much (a doorway is wider
                       # than this only when it is a real opening, which splits the face anyway)
SNAP_DEG = 20.0        # a normal this close to an axis is on it; further out is clustered by direction
PEAK_MIN = 0.60        # m2 of wall surface for a bin to be a wall face
OUTSIDE_GAP = 0.30     # m; a peak at least this far from the exterior face is a different wall
HORIZ_DEG = 25.0       # a wall vertex's own normal must be within this of horizontal
BEHIND = (0.15, 0.80)  # m behind a face: the strip tested for floor (interior) or nothing (exterior)
MAX_EXT = 1.5          # m; how far an interior wall may be extended at one end
PARALLEL_DEG = 20.0
DEFAULT_STOREY = 2.80
THICK_ASSUMED = {"exterior": 0.30, "interior": 0.15}
SIDE_NAMES = ("W", "S", "E", "N")     # side k has inward normal axes[k], as in close.py
MIN_KEYFRAME_PX = 2000
COVER_BIN = 0.10      # m; along-wall bins for "how much of the wall the key frames show"
MAX_VIEW_DEG = 75.0    # steeper views are not key frames (the harness producer's limit)
MIN_AGREEMENT = 0.5    # share of the wall's own vertices that must land inside the frame's mask


# ---------------------------------------------------------------- the annotated mesh
def gt_geometry():
    """Wall vertices and their normals in ScanNet's raw scan frame, plus the instance tables.

    The GT wall handoff is delivered in the frame the annotation lives in (`scannet_scan`, the
    frame of the mesh and of the GT poses), like the old `gt/` object handoff. Nothing is moved
    into amb3r_floor; the map into it is written alongside (`frame.to_amb3r_floor`).
    """
    P0, _rgb, F, owner, labels, _A = gsg.load(SCENE)
    VN = gsg.vertex_normals(P0, F)
    tri = P0[F]
    tri_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    return P0.astype(np.float64), VN, F, owner, labels, tri_area


def scan_to_floor() -> np.ndarray:
    """scannet_scan -> amb3r_floor (rigid), for overlaying this file on a harness handoff."""
    return scannet_scan_to_frame(Frame.load(), json.loads(DEFAULT_SIM3.read_text())["sim3_oracle_scale"])


def axes_of(VN, owner, labels):
    """The room's main axis (radians, mod 90) from vertical wall normals, weighted by count."""
    wall = np.isin(owner, [i for i, l in labels.items() if l == "wall"])
    h = wall & (np.abs(VN[:, 2]) < np.sin(np.radians(HORIZ_DEG)))
    n = VN[h, :2] / np.maximum(np.linalg.norm(VN[h, :2], axis=1, keepdims=True), 1e-9)
    theta = float(np.angle(np.exp(4j * np.arctan2(n[:, 1], n[:, 0])).sum()) / 4)
    return theta, [np.array([np.cos(theta + k * np.pi / 2), np.sin(theta + k * np.pi / 2)]) for k in range(4)], h, n


def face_peaks(P, owner, labels, tri_area, F, axes, vert, n2):
    """Wall faces as peaks in the offset histogram along each axis.

    A vertex goes to the axis its *own* normal points along (`kax`), not to every axis it is
    parallel to: `|n.e|` is the same for `e` and `-e`, so testing |n.e| would count each face
    under both k and k+2 and the same wall would come out twice. A vertex whose normal is not
    within 20 deg of any axis (a diagonal wall) is left out and counted.

    A fixed bin grid, not chained clustering: a chain walks across the room one bin at a
    time and joins walls that are metres apart (`walls.merge_lines` documents the same trap).

    Vertices off every axis are not thrown away. This room has one wall at 33.9 deg to the
    main axis - real surface, not corner noise - and dropping the off-axis vertices deleted it
    whole. But most off-axis vertices *are* corner noise: a vertex normal is the area-weighted
    average of its adjacent faces, so at a corner, a doorway edge or a wall top it points
    between two faces. The two are told apart by support, not by angle: off-axis vertices are
    grouped by their own direction, and a group becomes a wall only if some plane within it
    carries PEAK_MIN of surface. Noise spreads over every direction and never does; a real
    daigon walls does. Those faces come out with `axis = None`, as close.py treats its W5.
    """
    faces = []
    all_e = np.array(axes)
    kax = np.argmax(n2 @ all_e.T, axis=1)
    best = np.max(n2 @ all_e.T, axis=1)
    on_axis = best >= np.cos(np.radians(SNAP_DEG))
    for k, e in enumerate(axes):
        faces += _peaks(P, vert, np.flatnonzero((kax == k) & on_axis), e, k)
    # off-axis: cluster by direction - the full 360 deg, not mod 180. The two faces of a slanted
    # partition point opposite ways; grouped mod 180 they fell into one group and one peak, and the
    # face that came out was the two sides of the wall averaged, facing whichever side had more
    # vertices (here the bedroom side, so the bathroom wall was laid with its thickness into the
    # bathroom). Kept apart, each side is its own face and the pairing step below makes them one wall.
    off = np.flatnonzero(~on_axis)
    ang = np.arctan2(n2[off, 1], n2[off, 0])
    # Directions are the peaks of a 5 deg histogram over the full circle, and every vertex goes
    # to the nearest peak. Seeding groups greedily from whichever vertex comes first split one
    # face 12 deg wide into two groups, and two faces 3.8 cm apart came out as two walls.
    step = np.radians(5.0); nb = int(round(2 * np.pi / step))
    hist = np.bincount(((ang + np.pi) / step).astype(int) % nb, minlength=nb)
    smooth = sum(np.roll(hist, k) * w for k, w in zip((-2, -1, 0, 1, 2), (1, 2, 3, 2, 1)))
    cand = [i for i in range(nb) if smooth[i] >= PEAK_POINTS
            and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[(i + 1) % nb]]
    centres = []
    for i in sorted(cand, key=lambda i: -smooth[i]):
        a = -np.pi + (i + 0.5) * step
        if all(abs(np.angle(np.exp(1j * (a - b)))) >= np.radians(SNAP_DEG) for b in centres):
            centres.append(a)
    groups = [{"a": a, "idx": []} for a in centres]
    for i, a in zip(off, ang):
        if not centres:
            break
        dev = [abs(np.angle(np.exp(1j * (a - c)))) for c in centres]
        k = int(np.argmin(dev))
        if dev[k] < np.radians(SNAP_DEG):
            groups[k]["idx"].append(i)
    groups.sort(key=lambda g: -len(g["idx"]))
    for g in groups:
        if len(g["idx"]) < PEAK_POINTS:
            continue
        e = np.array([np.cos(g["a"]), np.sin(g["a"])])
        faces += _peaks(P, vert, np.array(g["idx"]), e, None)
    print(f"  {int((~on_axis).sum())} vertical wall vertices off every axis, in {len(groups)} directions; "
          f"{len(faces)} faces ({sum(1 for f in faces if f['axis'] is None)} diagonal)")
    return faces


def _peaks(P, vert, idx, e, axis):
    """One face per peak in the projection of P[vert][idx] onto direction e."""
    if len(idx) < 100:
        return []
    off = P[vert][idx, :2] @ e
    lo = np.floor(off.min() / BIN).astype(int)
    nb = np.floor(off.max() / BIN).astype(int) - lo + 1
    count = np.bincount(np.floor(off / BIN).astype(int) - lo, minlength=nb)
    smooth = np.convolve(count, [1, 2, 3, 2, 1], mode="same")
    peaks = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] < PEAK_POINTS or smooth[i] < smooth[i - 1] or smooth[i] < smooth[i + 1]:
            continue
        c = (lo + i + 0.5) * BIN
        sel = idx[np.abs(off - c) < 0.15]
        if len(sel) < 120:
            continue
        peaks.append((c, sel))
    # one face per peak: drop a peak within 0.25 m of a stronger one
    keep = []
    for c, sel in sorted(peaks, key=lambda t: -len(t[1])):
        if any(abs(c - c2) < 0.25 for c2, _ in keep):
            continue
        keep.append((c, sel))
    # The bin centre only says which 10 cm slab the face is in; where the face is, is the median
    # of its own vertices. Using the bin centre put walls up to 3.7 cm off, and by a different
    # amount in every frame, because the bin grid's phase depends on where the origin is.
    out = []
    u = np.array([-e[1], e[0]])
    for c, sel in sorted(keep):
        # A face is a wall where its vertices actually run along it. Corner vertices that happen
        # to point the same way can line up in one 10 cm slab from opposite ends of the room -
        # 143 vertices spread over 6.9 m once made a 'wall' across the whole room. Keep only the
        # longest stretch without a gap wider than GAP, and require it to carry the support.
        al = P[vert][sel, :2] @ u
        order = np.argsort(al)
        breaks = np.flatnonzero(np.diff(al[order]) > GAP)
        runs = np.split(order, breaks + 1)
        run = max(runs, key=len)
        if len(run) < PEAK_POINTS:
            continue
        sel = sel[run]
        out.append({"axis": axis, "offset": float(np.median(P[vert][sel, :2] @ e)), "idx": sel,
                    "pts": P[vert][sel], "normal": e.copy(), "n_sign": 1.0})
    return out


def free_run(face, others, limit=20.0) -> float:
    """Distance from the middle of `face`, along its own normal, to the first other wall face."""
    c = face["c"] + 0.5 * (face["a"][0] + face["a"][1]) * face["u"]
    d = face["n"]
    best = limit
    for o in others:
        p0 = o["c"] + o["a"][0] * o["u"]; e = (o["a"][1] - o["a"][0]) * o["u"]
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-9:
            continue
        w = p0 - c
        t = (w[0] * e[1] - w[1] * e[0]) / den            # along the ray
        s = (w[0] * d[1] - w[1] * d[0]) / den            # along the other face, 0..1
        if t > 0.05 and -0.05 <= s <= 1.05:
            best = min(best, t)
    return best


def refit(pts, n_hint):
    """Plane through pts with a horizontal normal, oriented to n_hint."""
    c = pts.mean(0)
    xy = pts[:, :2] - c[:2]
    _, v = np.linalg.eigh(np.cov(xy.T))
    n = v[:, 0] / max(np.linalg.norm(v[:, 0]), 1e-9)
    if n @ n_hint[:2] < 0:
        n = -n
    u = np.array([-n[1], n[0]])
    return c, n, u


# ---------------------------------------------------------------- main
def build(scene: Path = SCENE, output: Path = DEFAULT_OUTPUT) -> dict:
    P, VN, F, owner, labels, tri_area = gt_geometry()
    theta, axes, vert, n2 = axes_of(VN, owner, labels)
    faces = face_peaks(P, owner, labels, tri_area, F, axes, vert, n2)
    print(f"main axis {np.degrees(theta):+.2f} deg; {len(faces)} wall faces")

    floor_ids = [i for i, l in labels.items() if l == "floor"]
    floor_pts = P[np.isin(owner, floor_ids)]
    floor_xy = floor_pts[:, :2]
    floor_centre = floor_xy.mean(0)
    floor_z = float(np.median(floor_pts[:, 2]))
    # Heights in a wall handoff run from the floor: walls stand on z = floor and rise height_m.
    # In the scan frame the floor sits at floor_z (0.054 m here), so every z below is taken
    # relative to it; x and y stay scan-frame coordinates. `frame.floor_z` records the offset.
    P = P.copy(); P[:, 2] -= floor_z
    floor_pts = P[np.isin(owner, floor_ids)]

    ceiling_ids = [i for i, l in labels.items() if l == "ceiling"]
    ceil_pts = P[np.isin(owner, ceiling_ids)] if ceiling_ids else np.zeros((0, 3))
    if len(ceil_pts) > 200:
        storey = float(np.median(ceil_pts[:, 2]))
        storey_kind = "measured"
        storey_ev = {"ceiling_z_m": round(float(np.median(ceil_pts[:, 2]) + floor_z), 3),
                     "floor_z_m": round(floor_z, 3),
                     "ceiling_z_p05_p95": [round(float(np.percentile(ceil_pts[:, 2], 5)), 3),
                                           round(float(np.percentile(ceil_pts[:, 2], 95)), 3)],
                     "note": "median of the annotated ceiling vertices above the annotated floor"}
    else:
        storey, storey_kind, storey_ev = DEFAULT_STOREY, "assumed", {"why_not_measured": "no ceiling instance in the annotation"}

    # ---- exterior face per side: the peak furthest out on that axis
    ext, interior = {}, []
    for k in range(4):
        cand = [f for f in faces if f["axis"] == k]
        if not cand:
            raise ValueError(f"side {SIDE_NAMES[k]} has no wall face in the annotation")
        outer = min(cand, key=lambda f: f["offset"])
        ext[k] = outer
        interior += [f for f in cand if f is not outer and abs(f["offset"] - outer["offset"]) > OUTSIDE_GAP]
    # A face off every axis is never an exterior wall: the four rectangle sides are defined by
    # the four axes. It is an interior wall by elimination - this is close.py's `axis = None`.
    interior += [f for f in faces if f["axis"] is None]
    for k, f in ext.items():
        f["role"] = "exterior"
    for f in interior:
        f["role"] = "interior"

    # ---- orient each face and measure it
    records = []
    for f in faces:
        if f is not ext.get(f["axis"]) and f not in interior:
            f["role"] = "dropped"
            continue
        c, n, u = refit(f["pts"], f["normal"] * (1 if f["n_sign"] >= 0 else -1))
        # the normal keeps the surface's own facing: a partition's two faces then point
        # opposite ways and the pairing step below can recognise them as one wall. Flipping an
        # interior face towards "the side with more floor" cannot do that, because the two
        # faces of a 10 cm partition see almost the same floor and both flip the same way.
        d = (floor_xy - c[:2]) @ n
        f["room_floor_share"] = float((d > 0).mean())
        f["behind_floor"] = float(((np.abs(d) > BEHIND[0]) & (np.abs(d) < BEHIND[1])).mean())
        if f["role"] == "exterior" and d.mean() < 0:
            n, u = -n, -u                                  # the room is in front of an exterior wall
            f["room_floor_share"] = 1.0 - f["room_floor_share"]
        al = (f["pts"][:, :2] - c[:2]) @ u
        a = [float(np.percentile(al, 0.5)), float(np.percentile(al, 99.5))]
        dist = (f["pts"][:, :2] - c[:2]) @ n
        # f["idx"] indexes into the vertical-vertex subset (P[vert]), not into all vertices
        inst = [i for i in sorted({int(o) for o in owner[vert][f["idx"]]}) if labels.get(i) == "wall"]
        f.update({"c": c[:2], "n": n, "u": u, "a": a, "z": [float(np.percentile(f["pts"][:, 2], 0.5)),
                                                        float(np.percentile(f["pts"][:, 2], 99.5))],
                  "rms_cm": float(np.sqrt((dist ** 2).mean()) * 100), "wall_ids": inst,
                  "frames": [], "masks": [], "behind_seen": f["behind_floor"],
                  "heading_deg": float(np.degrees(np.arctan2(n[1], n[0]))),
                  "length_m": a[1] - a[0], "point": c[:2].tolist(), "along": u.tolist()})
        f["plane_id"] = f"P{len(records) + 1}"
        records.append(f)

    # ---- a partition seen from both sides is one wall, not two
    # The annotation has both faces of an interior partition (and both sides of the
    # bathroom's south wall), so two opposite-facing planes 3-40 cm apart that overlap along
    # their length are the two sides of one wall. The wall is described from the face with
    # the room in front of it; the gap between them is its measured thickness.
    pairs, merged_faces = [], set()
    for i, f in enumerate(records):
        if f["role"] != "interior" or f["plane_id"] in merged_faces:
            continue
        for g in records[i + 1:]:
            if g["role"] != "interior" or g["plane_id"] in merged_faces:
                continue
            if abs(f["n"] @ g["n"]) < np.cos(np.radians(20)) or f["n"] @ g["n"] > 0:
                continue
            gap = abs(float((g["c"] - f["c"]) @ f["n"]))
            if not (0.03 < gap < 0.40):
                continue
            if not ((f["a"][0] < g["a"][1]) and (g["a"][0] < f["a"][1])):
                continue
            # Which face describes the wall (its room side; thickness goes behind it)? The one
            # facing the larger space: a partition between a bedroom and a bathroom is described
            # from the bedroom. Measured as the free run from each face, along its own normal, to
            # the first other wall. All faces of one partition chain then agree on which side the
            # thickness goes, which is what lets their corners mitre cleanly. (The floor share
            # beside each face, used before, is the same on both sides of a 10 cm wall and so
            # picked a side at random - and a chain whose sides disagree mitres into a wedge.)
            others = [h for h in records if h is not f and h is not g]
            first = f if free_run(f, others) >= free_run(g, others) else g
            other = g if first is f else f
            # Measure across the two faces' own vertices, inside the stretch where both exist,
            # rather than between two fitted plane offsets: a face that bends pools its straight
            # and slanted runs into one plane, so its offset is not where either run really is.
            a1 = (first["pts"][:, :2] - first["c"]) @ first["u"]
            a2 = (other["pts"][:, :2] - first["c"]) @ first["u"]
            lo = max(float(np.percentile(a1, 5)), float(np.percentile(a2, 5)))
            hi = min(float(np.percentile(a1, 95)), float(np.percentile(a2, 95)))
            sel = (a2 > lo) & (a2 < hi)
            sel_back = (a1 > lo) & (a1 < hi)
            if sel.sum() < 30 or sel_back.sum() < 30:
                continue
            d = (other["pts"][sel, :2] - first["c"]) @ first["n"]
            d_back = (first["pts"][sel_back, :2] - other["c"]) @ other["n"]
            thick = float(np.median(d))
            iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
            # The gap measured one way and the other must agree: on two parallel faces it does
            # exactly. They disagree when a face bends, because its own vertices are not all on
            # the plane being projected onto. That asymmetry is the cheapest parallel test there
            # is, and it needs no fitting.
            asym = abs(abs(float(np.median(d))) - abs(float(np.median(d_back))))
            first["thickness_measured"] = abs(thick)
            first["thickness_other"] = other["plane_id"]
            first["paired_with"] = other["plane_id"]
            first["thickness_angle_deg"] = float(np.degrees(np.arccos(np.clip(-(first["n"] @ other["n"]), -1.0, 1.0))))
            # When is the gap a wall's thickness? Only when the two faces describe one flat
            # wall over most of their length, and measure the same both ways round. A short
            # overlap means the faces were cut into different pieces and only their tails meet -
            # which is what happens on this partition, whose straight and slanted runs land in
            # different direction groups. The number is then a hint, not a measured thickness.
            span = float(min(first["length_m"], other["length_m"]))
            first["thickness_trusted"] = bool((hi - lo) >= max(0.30, 0.5 * span) and iqr < 0.06
                                              and asym < 0.02)
            first["thickness_overlap_m"] = round(hi - lo, 3)
            first["thickness_iqr_cm"] = round(iqr * 100, 1)
            first["thickness_asym_cm"] = round(asym * 100, 1)
            merged_faces.add(other["plane_id"])
            pairs.append((first, other))
            break
    records = [f for f in records if f["plane_id"] not in merged_faces]

    for i, f in enumerate(records):
        f["id"] = f"W{i + 1}"
    ext_rec = {f["axis"]: f for f in records if f["role"] == "exterior"}
    print(f"exterior {len(ext_rec)}; interior {sum(1 for f in records if f['role'] == 'interior')}"
          f" ({len(merged_faces)} opposite faces merged into their partner)")

    # ---- rectangle (same construction as close.py step 4)
    sides = []
    for k in range(4):
        f = ext_rec[k]
        sides.append({"k": k, "side": SIDE_NAMES[k], "normal": axes[k].tolist(), "offset": f["offset"],
                      "from": "observed", "walls": [f["id"]]})
    s0, t0 = sides[0]["offset"], sides[1]["offset"]
    s1, t1 = -sides[2]["offset"], -sides[3]["offset"]
    ex, ey = axes[0], axes[1]
    corners = [s0 * ex + t0 * ey, s1 * ex + t0 * ey, s1 * ex + t1 * ey, s0 * ex + t1 * ey]
    seg_of = {0: (corners[3], corners[0]), 1: (corners[0], corners[1]),
              2: (corners[1], corners[2]), 3: (corners[2], corners[3])}
    for sd in sides:
        p0, p1 = seg_of[sd["k"]]
        sd.update({"p0": p0.tolist(), "p1": p1.tolist(), "length_m": float(np.linalg.norm(p1 - p0)),
                   "observed": [[0.0, float(np.linalg.norm(p1 - p0))]],
                   "observed_m": float(np.linalg.norm(p1 - p0))})
    rect = {"corners": [c.tolist() for c in corners], "width_m": float(s1 - s0), "depth_m": float(t1 - t0),
            "area_m2": float((s1 - s0) * (t1 - t0)), "axis_deg": float(np.degrees(theta))}
    # What the annotation says about the rectangle, in the annotation's own terms. These are
    # not close.py's checks under different names: close.py asks the rays how much empty space
    # falls outside the rectangle; the annotation has no rays, and its exterior wall vertices
    # sit *behind* the room-side face by construction, so "outside" would be 100% by definition
    # of a wall with any thickness. The honest questions here are: how much of the rectangle has
    # annotated floor in it, and how far the wall vertices sit from the rectangle's own sides.
    CELL = 0.10
    cw = int(np.ceil((s1 - s0) / CELL)) + 1
    ch = int(np.ceil((t1 - t0) / CELL)) + 1
    floor_ij = np.stack([np.clip((floor_xy @ ex - s0) / CELL, 0, cw - 1),
                         np.clip((floor_xy @ ey - t0) / CELL, 0, ch - 1)], 1).astype(int)
    covered = np.zeros((cw, ch), bool)
    covered[floor_ij[:, 0], floor_ij[:, 1]] = True
    covered = ndimage.binary_dilation(covered, np.ones((3, 3), bool))
    wall_px = P[vert][:, :2]
    checks = {"measured_on": "the annotation mesh (close.py asks the scene rays instead)",
              "floor_covered_share": round(float(covered.mean()), 3),
              "wall_vertices_outside_share": round(float(((wall_px @ ex < s0 - 0.05) | (wall_px @ ex > s1 + 0.05)
                                                          | (wall_px @ ey < t0 - 0.05) | (wall_px @ ey > t1 + 0.05)).mean()), 4),
              "note": "wall_vertices_outside_share is not a defect: an exterior wall's own vertices sit "
                      "behind its room-side face, which is what the rectangle follows"}

    # ---- thickness: a partition has two faces; measure the gap where both exist
    for f in records:
        f.setdefault("thickness_measured", None)
        if f["role"] != "interior":
            continue
        for g in records:
            if g is f or g["role"] != "interior":
                continue
            # the two faces of a partition point opposite ways, so they land on opposite axes
            # (k and k+2); requiring the same axis would never pair them
            if abs((f["n"] @ g["n"])) < np.cos(np.radians(20)) or f["n"] @ g["n"] > 0:
                continue
            gap = abs(float((g["c"] - f["c"]) @ f["n"]))
            if not (0.03 < gap < 0.40):
                continue
            ov = (f["a"][0] < g["a"][1]) and (g["a"][0] < f["a"][1])
            if not ov:
                continue
            if f["thickness_measured"] is None or gap < f["thickness_measured"]:
                f["thickness_measured"] = gap
                f["thickness_other"] = g["id"]
                g.setdefault("thickness_measured", gap)
                g["thickness_other"] = f["id"]

    # ---- openings from the annotated instances
    openings_of, openings_skipped = openings(records, P, owner, labels, rect)

    # ---- extend interior walls to the first line they meet
    lines = [{"id": f"side{sd['k']}", "c": np.array(sd["p0"]),
              "n": np.array(sd["normal"]),
              "u": (np.array(sd["p1"]) - np.array(sd["p0"])) / sd["length_m"], "a": [0.0, sd["length_m"]]}
             for sd in sides]
    lines += [{"id": f["id"], "c": f["c"], "n": f["n"], "u": f["u"], "a": f["a"]}
              for f in records if f["role"] == "interior"]
    for f in records:
        f["ends"] = []
        if f["role"] != "interior":
            continue
        for sgn, a_end in ((-1, f["a"][0]), (1, f["a"][1])):
            p = f["c"] + a_end * f["u"]
            dv = sgn * f["u"]
            best = None
            for M in lines:
                if M["id"] == f["id"]:
                    continue
                den = dv @ M["n"]
                if abs(den) < np.sin(np.radians(PARALLEL_DEG)):
                    continue
                t = float((M["c"] - p) @ M["n"] / den)
                if not (-0.3 <= t <= MAX_EXT):
                    continue
                h = p + t * dv
                ah = float((h - M["c"]) @ M["u"])
                slack = 0.05 if M["id"].startswith("side") else MAX_EXT
                if not (M["a"][0] - slack <= ah <= M["a"][1] + slack):
                    continue
                if best is None or t < best[0]:
                    best = (t, M["id"], h)
            end = {"side": "start" if sgn < 0 else "end", "from": p.tolist()}
            if best is None:
                end.update({"to": p.tolist(), "ext_m": 0.0, "meets": None,
                            "why": f"no wall line within {MAX_EXT} m"})
            else:
                t, other, h = best
                end.update({"to": h.tolist(), "ext_m": t, "meets": other})
            f["ends"].append(end)

    # ---- final walls, in close.py's shape so build_walls() can consume them
    final = []
    for sd in sides:
        f = ext_rec[sd["k"]]
        p0 = np.array(sd["p0"]); dv = (np.array(sd["p1"]) - p0) / sd["length_m"]
        final.append({"id": f"ext_{sd['side']}", "side": sd["k"], "role": "exterior",
                      "walls": [f["id"]], "normal": sd["normal"], "position_from": "observed",
                      "pieces": [{"p0": p0.tolist(), "p1": np.array(sd["p1"]).tolist(), "kind": "observed"}]})
    for f in records:
        if f["role"] != "interior":
            continue
        a0 = f["c"] + f["a"][0] * f["u"]; a1 = f["c"] + f["a"][1] * f["u"]
        e0, e1 = f["ends"]
        pieces = []
        if e0["ext_m"] > 0.01:
            pieces.append({"p0": e0["to"], "p1": a0.tolist(), "kind": "inferred"})
        if e0["ext_m"] < -0.01:
            a0 = np.array(e0["to"])
        pieces.append({"p0": a0.tolist(), "p1": a1.tolist(), "kind": "observed"})
        if e1["ext_m"] > 0.01:
            pieces.append({"p0": a1.tolist(), "p1": e1["to"], "kind": "inferred"})
        final.append({"id": f["id"], "role": "interior", "walls": [f["id"]],
                      "normal": f["n"].tolist(), "pieces": pieces})
    for fl in final:
        for p in fl["pieces"]:
            p["length_m"] = float(np.linalg.norm(np.array(p["p1"]) - np.array(p["p0"])))
        fl["length_m"] = sum(p["length_m"] for p in fl["pieces"])
        fl["inferred_m"] = sum(p["length_m"] for p in fl["pieces"] if p["kind"] == "inferred")

    # ---- thickness per wall
    for fl in final:
        src = next(f for f in records if f["id"] == fl["walls"][0])
        m = src.get("thickness_measured")
        hint = {"kind": "two_faces", "other_plane": src.get("thickness_other"),
                "gap_cm": round((m or 0.0) * 100, 1),
                "angle_deg": round(float(src.get("thickness_angle_deg", 0.0)), 1),
                "overlap_m": src.get("thickness_overlap_m"),
                "spread_cm": src.get("thickness_iqr_cm")}
        if m is not None and src.get("thickness_trusted"):
            fl["thickness_m"] = float(round(m, 3)); fl["thickness_kind"] = "measured"
            fl["thickness_hint"] = [hint]
        elif m is not None:
            # the other face was seen, but the pair does not describe one flat wall: keep the
            # hint, do not hand the number over as measured
            fl["thickness_m"] = THICK_ASSUMED[fl["role"]]; fl["thickness_kind"] = "assumed"
            hint["why_not_measured"] = (
                f"both faces were seen, but they overlap over only {src.get('thickness_overlap_m')} m "
                f"and their gap spreads {src.get('thickness_iqr_cm')} cm. This partition bends, so its "
                "straight and slanted runs were fitted as separate faces and only their tails meet; "
                "the gap between what is left is not one wall's thickness. Read "
                "`thickness_defaults.interior_m` instead, and treat this as a hint")
            fl["thickness_hint"] = [hint]
        else:
            fl["thickness_m"] = THICK_ASSUMED[fl["role"]]; fl["thickness_kind"] = "assumed"
            fl["thickness_hint"] = []

    # ---- the walls.json-shaped plane table build_walls() reads
    plane_rows = []
    for f in records:
        row = {"id": f["id"], "point": f["point"], "normal": f["n"].tolist(), "along": f["along"],
               "a": f["a"], "z": f["z"], "area_m2": float(len(f["idx"])), "frames": [], "masks": [],
               "objects": f["wall_ids"],
               "rms_cm": f["rms_cm"], "heading_deg": f["heading_deg"],
               "snap_deg": 0.0, "behind_seen": f["behind_floor"], "wall_ids": f["wall_ids"],
               "openings": openings_of.get(f["id"], [])}
        plane_rows.append(row)

    closed = {"storey": {"height_m": storey, "kind": storey_kind, "evidence": storey_ev,
                         "observed_wall_top_m": float(max(f["z"][1] for f in records))},
              "thickness": {"exterior_m": THICK_ASSUMED["exterior"], "interior_m": THICK_ASSUMED["interior"],
                            "kind": "assumed"},
              "axis_deg": float(np.degrees(theta)), "rectangle": rect, "sides": sides, "checks": checks,
              "walls": [{**{k: v for k, v in f.items() if k not in ("pts", "idx")},
                         "openings": openings_of.get(f["id"], [])} for f in records],
              "final": final}
    planes = {"walls": plane_rows, "rejected": []}

    walls = build_walls(closed, planes)
    # build_walls() hard-codes thickness_kind="assumed" (the harness producer always assumes
    # it) and renames interior walls to int_N; here a partition measured on both faces is not
    # an assumption, so put the measured value back. Its output order follows closed["final"].
    assert len(walls) == len(final), "build_walls dropped a wall"
    for w, fl in zip(walls, final):
        w["thickness_m"], w["thickness_kind"] = fl["thickness_m"], fl["thickness_kind"]
        w["evidence"]["thickness_hint"] = fl.get("thickness_hint", [])
        # build_walls() writes every opening with source "ray_test" and drops the instance it
        # came from. These openings were not found by rays, so say what they are.
        src_of = {(round(o["a"][0], 3), round(o["a"][1], 3)): o
                  for o in openings_of.get(fl["walls"][0], [])}
        for o in w["openings"]:
            m = next((v for k, v in src_of.items()
                      if abs(k[0] - 0) < 1e9 and abs(o["to_m"] - o["from_m"] - (k[1] - k[0])) < 0.05), None)
            o["source"] = "gt_instance"
            if m is not None:
                o["instance"] = m["instance"]
                o["note"] = ("an annotated doorway with no door leaf: this is a hole a consumer may walk "
                             "or look through. A door leaf or a window elsewhere on this wall is its own "
                             "item and gets no hole - see `openings_not_taken`")
    print(f"{len(walls)} walls")

    # ---- key frames from the annotated 2D instance masks
    out = output.resolve()
    camera = None
    meta = read_scene_meta(scene)
    camera = schema.camera(meta["colorWidth"], meta["colorHeight"], meta["fx_color"], meta["fy_color"],
                           meta["mx_color"], meta["my_color"],
                           source=f"{scene / (scene.name + '.txt')} (fx_color, fy_color, mx_color, my_color, "
                                  "colorWidth, colorHeight); the same intrinsics as the object handoff")
    # a rerun must not leave a previous run's wall directories behind (wall ids change when
    # the face grouping changes, and walls_check reads every directory under keyframes/)
    import shutil as _shutil
    if (out / "keyframes").is_dir():
        _shutil.rmtree(out / "keyframes")
    keys, kf_stats = key_frames(walls, records, camera, scene, out, floor_z=floor_z)
    for w in walls:
        w["key_frames"] = keys.get(w["id"], [])
        st = kf_stats.get(w["id"], {})
        w["evidence"]["key_frame_candidates"] = st.get("candidates", 0)
        w["evidence"]["key_frames_show"] = st.get("show", {"observed_share": 0.0, "wall_share": 0.0})

    closed["storey"]["evidence"]["observed_wall_top_m"] = closed["storey"]["observed_wall_top_m"]
    doc = assemble(walls, closed, camera, run="gt", scene=scene, storey=storey,
                   openings_skipped=openings_skipped)
    out.mkdir(parents=True, exist_ok=True)
    (out / "walls.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    cam_xy = {(w["id"], k["source_frame_id"]): np.array(k["camera_to_world"])[:2, 3]
              for w in walls for k in w["key_frames"]}
    preview(doc, cam_xy, out / "preview.png")
    (out / "README.md").write_text(readme(doc, out, scene))
    print(f"wrote {out}")
    for w in walls:
        print(f"  {w['id']:6} {w['role']:8} {w['length_m']:.2f} m  {w['thickness_kind']:8} "
              f"{w['thickness_m']:.2f} m  openings {len(w['openings'])}  key frames {len(w['key_frames'])}")
    return doc


# ---------------------------------------------------------------- openings
def openings(records, P, owner, labels, rect):
    """Door frames, doors and windows, expressed on the wall they hang on."""
    item_labels = ("door", "doorframe", "window")
    items = []
    for oid, lab in sorted(labels.items()):
        if lab not in item_labels or not (owner == oid).any():
            continue
        q = P[owner == oid]
        best, host = 1e9, None
        for f in records:
            c, n, u = f["c"], f["n"], f["u"]
            d = float(np.median((q[:, :2] - c) @ n))
            al = (q[:, :2] - c) @ u
            ov = ((al > f["a"][0] - 0.4) & (al < f["a"][1] + 0.4)).mean()
            # the item must lie on the face's plane, not merely project onto its line
            if ov < 0.6 or abs(d) > 0.45:
                continue
            if abs(d) < best:
                best, host = abs(d), f
        if host is None:
            continue
        c, n, u = host["c"], host["n"], host["u"]
        d = float(np.median((q[:, :2] - c) @ n))
        al = (q[:, :2] - c) @ u
        items.append({"id": oid, "label": lab, "host": host["id"], "offset_m": d,
                      "a": [float(al.min()), float(al.max())],
                      "z": [float(np.percentile(q[:, 2], 1)), float(np.percentile(q[:, 2], 99))]})

    out = {}
    by_host = {}
    for it in items:
        by_host.setdefault(it["host"], []).append(it)
    host_of = {f["id"]: f for f in records}
    skipped = []
    for host, group in by_host.items():
        frames = [i for i in group if i["label"] == "doorframe"]
        leaves = [i for i in group if i["label"] == "door"]
        wins = [i for i in group if i["label"] == "window"]
        hw = host_of[host]
        # Only a doorway with no leaf is handed over as an opening. A door leaf and a window are
        # their own items - `not_included` says so - and the wall generator must not cut a hole
        # for them: a closed leaf fills its frame and a window is a pane, not a gap. What this
        # file promises a wall has is the holes you can walk or look through with nothing in them.
        for w in wins:
            skipped.append({"instance": f"window#{w['id']}", "wall": host,
                            "why": "a window is its own item, not a gap in the wall (see not_included)"})
        for fr in frames:
            mate = next((d for d in leaves
                         if d["a"][0] < fr["a"][1] and fr["a"][0] < d["a"][1]), None)
            if mate is not None:
                skipped.append({"instance": f"doorframe#{fr['id']}", "wall": host,
                                "why": f"a door leaf (#{mate['id']}) hangs in this frame, so the "
                                       "frame is a closure, not an opening"})
                continue
            out.setdefault(host, []).append(hole_from(fr, "doorway", inferred=False, host=hw))
        framed = [fr for fr in frames]
        for d in leaves:
            if any(d["a"][0] < fr["a"][1] and fr["a"][0] < d["a"][1] for fr in framed):
                continue
            skipped.append({"instance": f"door#{d['id']}", "wall": host,
                            "why": "a door leaf is its own item; without an annotated frame round "
                                   "it the hole it fills is not measured"})
    return out, skipped


def hole_from(item, what, inferred, host):
    """One opening record in the shape walls/v2 wants, from an annotated instance.

    `corners` are the four corners on the wall's own plane, in the order the contract uses
    (start-low, end-low, end-high, start-high); `build_walls` reads the first two to place the
    opening along the wall.
    """
    a0, a1 = item["a"]
    z0 = 0.0 if item["z"][0] < 0.15 else float(round(item["z"][0], 3))
    z1 = float(round(item["z"][1], 3))
    c, u, d = host["c"], host["u"], item["offset_m"]
    n = host["n"]
    quad = [[*(c + a * u + d * n), z] for a, z in ((a0, z0), (a1, z0), (a1, z1), (a0, z1))]
    return {"a": [a0, a1], "z": [z0, z1], "width_m": round(a1 - a0, 3), "height_m": round(z1 - z0, 3),
            "rays": 0, "closed_above": True, "seen_bottom_m": float(round(item["z"][0], 3)),
            "corners": [[float(x) for x in q] for q in quad],
            "bounds": {"left": "solid", "right": "solid", "top": "solid",
                       "bottom": "floor" if z0 == 0.0 else "solid"},
            "what": what, "instance": item["label"] + "#" + str(item["id"]),
            "inferred": bool(inferred), "host": host["id"]}


# ---------------------------------------------------------------- key frames
def key_frames(walls, records, camera, scene: Path, out: Path, floor_z: float = 0.0):
    """Per wall, up to three key frames; also the per-wall statistics the page shows."""
    from .from_gt import frame_clarity
    clarity_cache: dict[int, float] = {}

    def clarity_of(fid: int) -> float:
        if fid not in clarity_cache:
            p = FRAMES / f"frame_{fid:06d}.jpg"
            clarity_cache[fid] = frame_clarity(p) if p.is_file() else 0.0
        return clarity_cache[fid]

    """Per wall, up to three frames whose 2D instance mask shows it, scored by visible area."""
    K = np.asarray(camera["K"], dtype=np.float64)
    W, H = int(camera["width"]), int(camera["height"])
    source = json.loads((SOURCE / "manifest.json").read_text())
    seq_of = [(int(f["sequence_index"]), int(f["source_frame_id"])) for f in source["frames"]]
    poses = {}
    for seq, _fid in seq_of:
        p = GT_INPUT / f"{seq:04d}.npz"
        if p.is_file():
            with np.load(p) as z:
                if "camera_pose" in z.files:
                    poses[seq] = z["camera_pose"].astype(np.float64)
    # GT poses are scan-frame; lower them by the floor offset the geometry was lowered by
    S = np.eye(4); S[2, 3] = -floor_z

    # clarity is a whole-frame measure; the page reads it relative to the median frame, as the
    # harness producer's key frames do
    clarity_all = [clarity_of(fid) for _seq, fid in seq_of]
    clarity_all = [c for c in clarity_all if c > 0]
    clarity_median = float(np.median(clarity_all)) if clarity_all else 1.0

    inst_of = {f["id"]: set(f["wall_ids"]) for f in records}
    pts_of = {f["id"]: f["pts"] for f in records}
    cand = {w["id"]: [] for w in walls}
    zpath = scene / f"{scene.name}_2d-instance.zip"
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    for w in walls:
        (out / "keyframes" / w["id"]).mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath) as archive:
        names = {int(n.split("/")[-1][:-4]): n for n in archive.namelist() if n.endswith(".png")}
        for seq, fid in seq_of:
            if fid not in names or seq not in poses:
                continue
            raw = np.asarray(Image.open(io.BytesIO(archive.read(names[fid]))))
            if raw.shape != (H, W):
                continue
            pose = S @ poses[seq]
            cam = pose[:3, 3]
            for w in walls:
                ids = inst_of[w["_planes"][0]["id"]] if w["_planes"] else set()
                m = np.isin(raw.astype(np.int32) - 1, list(ids)) if ids else None
                if m is None or m.sum() < MIN_KEYFRAME_PX:
                    continue
                p0 = np.asarray(w["start_floor"])          # (2,) on z = 0
                d = w["_dir"]
                room = -np.asarray(w["normal_to_back"])     # (2,)
                p0_3 = np.array([p0[0], p0[1], 0.0])
                d3 = np.array([d[0], d[1], 0.0])
                # every 4th mask pixel -> ray -> wall plane -> along the wall
                ys, xs = np.nonzero(m)
                take = np.arange(0, len(xs), 4)
                X = np.stack([xs[take] + 0.5, ys[take] + 0.5, np.ones(len(take))], 1) @ np.linalg.inv(K).T
                Xw = X @ pose[:3, :3].T + pose[:3, 3]
                room3 = np.array([room[0], room[1], 0.0])
                # ray from the camera centre through each pixel: cam + lam (Xw - cam); it meets the
                # wall plane where (cam + lam (Xw - cam) - p0) . n = 0
                ray = Xw - cam[None, :]
                den = ray @ room3
                lam = ((p0_3 - cam) @ room3) / np.where(np.abs(den) < 1e-9, 1e-9, den)
                hit = cam[None, :] + lam[:, None] * ray
                al = (hit - p0_3) @ d3
                cover = [float(np.clip(np.percentile(al, 2), 0, w["length_m"])),
                         float(np.clip(np.percentile(al, 98), 0, w["length_m"]))]
                centroid = p0_3 + 0.5 * w["length_m"] * d3
                v = centroid[:2] - cam[:2]; dist = float(np.linalg.norm(v))
                view = float(np.degrees(np.arccos(np.clip(-(v / max(dist, 1e-9)) @ room[:2], -1, 1))))
                # a view from behind the wall is not a view of its room-side face
                if view > MAX_VIEW_DEG:
                    continue
                ys_, xs_ = np.nonzero(m)
                box = [int(xs_.min()), int(ys_.min()), int(xs_.max()) + 1, int(ys_.max()) + 1]

                def dist_to_mask(ui, vi):
                    """Pixels from each (u, v) to the nearest pixel inside this wall's 2D mask."""
                    if not hasattr(dist_to_mask, "_cache") or dist_to_mask._cache[0] is not m:
                        dist_to_mask._cache = (m, ndimage.distance_transform_edt(~m))
                    dmap = dist_to_mask._cache[1]
                    return dmap[np.clip(vi, 0, dmap.shape[0] - 1), np.clip(ui, 0, dmap.shape[1] - 1)]
                # wall surface this frame shows, 3 cm voxels on the plane (the same stand-in for
                # area the wall pipeline uses); depth offset of those points from the file's plane
                cells = np.unique(np.floor(hit / 0.03).astype(np.int64), axis=0)
                area = float(len(cells) * 0.03 ** 2)
                offset = float(np.median(hit @ room3 - p0_3 @ room3))
                # How well this frame's pose and intrinsics agree with the annotation: push the
                # wall's own mesh vertices through them and measure where they land relative to
                # the 2D mask the annotation gives that wall. `reprojection_px` is the median
                # distance in pixels from a projected vertex to the nearest mask pixel (0 inside);
                # `mask_agreement` is the share that lands inside. A pose or intrinsics error
                # shows up here as a large median or a low share.
                Wp = pts_of.get(w["_planes"][0]["id"]) if w["_planes"] else None
                if Wp is not None and len(Wp):
                    step = max(1, len(Wp) // 3000)
                    Q = Wp[::step]
                    Tinv = np.linalg.inv(pose)                 # world -> camera
                    Qc = Q @ Tinv[:3, :3].T + Tinv[:3, 3]
                    okq = Qc[:, 2] > 0.05
                    uu = K[0, 0] * Qc[okq, 0] / Qc[okq, 2] + K[0, 2]
                    vv = K[1, 1] * Qc[okq, 1] / Qc[okq, 2] + K[1, 2]
                    inimg = (uu >= 0) & (uu < W - 1) & (vv >= 0) & (vv < H - 1)
                    if inimg.sum() >= 20:
                        ui = np.round(uu[inimg]).astype(int); vi = np.round(vv[inimg]).astype(int)
                        dist_px = dist_to_mask(ui, vi)
                        reproj = float(np.median(dist_px))
                        agreement = float((dist_px <= 1.0).mean())
                    else:
                        reproj, agreement = 9999.0, 0.0
                else:
                    reproj, agreement = 9999.0, 0.0
                # The wall's instance mask in this frame may belong to a *different* face of the
                # same instance (a corner, the far side of a partition): then the wall's own
                # vertices do not land on it. Such a frame does not show this wall.
                if agreement < MIN_AGREEMENT or area < 0.05:
                    continue
                cand[w["id"]].append({
                    "seq": seq, "source_frame_id": fid, "mask": m, "pixels": int(m.sum()),
                    "cover": cover, "view": view, "dist": dist, "pose": pose,
                    "area": area, "offset": offset, "reproj": reproj, "agreement": agreement,
                    "clarity": clarity_of(fid),
                    "bearing": (centroid - cam) / max(np.linalg.norm(centroid - cam), 1e-9),
                    "touches_border": bool(box[0] == 0 or box[1] == 0 or box[2] == W or box[3] == H),
                    "bbox": box, "score": float(m.sum() / m.size)})
    keys, stats = {}, {}
    for w in walls:
        entries = cand[w["id"]]
        chosen = select_keyframes(entries) if entries else []
        chosen.sort(key=lambda c: -c["pixels"])
        recs = []
        for i, c in enumerate(chosen[:ws.MAX_KEYFRAMES]):
            stem = f"keyframes/{w['id']}/{c['seq']:04d}"
            import shutil
            shutil.copyfile(FRAMES / f"frame_{c['source_frame_id']:06d}.jpg", out / f"{stem}_rgb.jpg")
            Image.fromarray((c["mask"] * 255).astype(np.uint8), "L").save(out / f"{stem}_mask.png")
            draw_boxed(Image.open(out / f"{stem}_rgb.jpg").convert("RGB"), c["bbox"], f"{w['id']} {schema.KEYFRAME_ROLES[i]}").save(
                out / f"{stem}_boxed.jpg", quality=95, subsampling=0)
            recs.append({"role": schema.KEYFRAME_ROLES[i], "seq": c["seq"], "source_frame_id": c["source_frame_id"],
                         "rgb": f"{stem}_rgb.jpg", "mask": f"{stem}_mask.png", "boxed": f"{stem}_boxed.jpg",
                         "bbox_xyxy": c["bbox"], "bbox_convention": schema.BBOX_CONVENTION,
                         "visible_pixels": c["pixels"],
                         "covers_m": [round(c["cover"][0], 3), round(c["cover"][1], 3)],
                         "visible_area_m2": round(c["area"], 3),
                         # no reprojection_px: this handoff carries GT poses, so projecting the
                         # mesh back into a frame is exact by construction and the number would
                         # say nothing. `mask_agreement` is the honest one - the share of this
                         # wall's vertices that land inside the annotation's own 2D mask.
                         "mask_agreement": round(c["agreement"], 3),
                         # no plane_offset_m: the mesh *is* the wall, there is no per-frame depth
                         # disagreement to report (the harness file reports AMB3R's)
                         "clarity": round(c["clarity"] / max(clarity_median, 1e-9), 2),
                         "view_angle_deg": round(c["view"], 1), "distance_m": round(c["dist"], 3),
                         "camera_to_world": [[round(float(x), 8) for x in r] for r in c["pose"]]})
        keys[w["id"]] = recs
        # how much of the wall the chosen frames show, in 10 cm bins along its length
        covered = np.zeros(max(1, int(np.ceil(w["length_m"] / COVER_BIN))), bool)
        for c in chosen[:ws.MAX_KEYFRAMES]:
            lo_b = int(np.floor(c["cover"][0] / COVER_BIN))
            hi_b = int(np.ceil(c["cover"][1] / COVER_BIN))
            covered[max(0, lo_b):max(0, min(len(covered), hi_b))] = True
        stats[w["id"]] = {"candidates": len(entries),
                          "show": {"observed_share": round(float(covered.mean()), 3),
                                   "wall_share": round(float(covered.mean()), 3)}}
    return keys, stats


# ---------------------------------------------------------------- the document
def scan_frame_block(floor_z: float) -> dict:
    """What `scannet_scan` means in this file, and how to get from it to amb3r_floor."""
    T = scan_to_floor()
    L = np.eye(4); L[2, 3] = floor_z                    # file coordinates -> raw scan frame
    return {
        "name": "scannet_scan",
        "definition": ("ScanNet's raw scan frame - the frame of `<scene>_vh_clean_2.ply` and of the GT "
                       "camera poses, before axisAlignment - in metres, +Z up. x and y are scan-frame "
                       "coordinates as they are; z is measured from the annotated floor, so a raw "
                       "scan-frame point is (x, y, z + floor_z)"),
        "floor_z": round(float(floor_z), 4),
        "floor_z_definition": "median z of the annotated `floor` instance's vertices, raw scan frame",
        "axis_alignment_applied": False,
        "to_amb3r_floor": {
            "matrix": [[round(float(v), 12) for v in row] for row in (T @ L)],
            "maps": "this file's coordinates -> amb3r_floor (p_floor = M @ [x, y, z, 1]); the same map "
                    "gt_amb3r_floor/ was made with, for overlaying on a harness handoff",
            "use": "evaluation and overlay only"},
    }


def assemble(walls, closed, camera, run, scene, storey, openings_skipped=()):
    def keep(w):
        return {k: v for k, v in w.items() if not k.startswith("_")}

    return {
        "schema_version": ws.SCHEMA_VERSION, "scene_id": scene.name,
        "producer": {"name": "scannet_ground_truth", "run": run,
                     "note": "walls measured on the ScanNet annotation mesh, not a scene-understanding run; "
                             "openings are the annotated doorways that have no door leaf (source gt_instance), "
                             "not holes the rays saw",
                     "sources": {"mesh": str(scene / f"{scene.name}_vh_clean_2.ply"),
                                 "aggregation": str(scene / f"{scene.name}.aggregation.json"),
                                 "instance_2d": str(scene / f"{scene.name}_2d-instance.zip"),
                                 "rgb": str(FRAMES)}},
        "coordinate_frame": "scannet_scan", "units": "meters", "up_axis": "Z",
        "frame": scan_frame_block(closed["storey"]["evidence"]["floor_z_m"]), "camera": camera,
        "storey": {"floor_z_floor": 0.0, "height_m": storey, "kind": closed["storey"]["kind"],
                   "evidence": closed["storey"]["evidence"]},
        "thickness_defaults": {"exterior_m": THICK_ASSUMED["exterior"],
                               "interior_m": THICK_ASSUMED["interior"], "kind": "assumed",
                               "applied": "behind the room-side face, along normal_to_back"},
        "outline": {"shape": "rectangle", "axis_deg": round(closed["axis_deg"], 3),
                    "inner_corners_floor": closed["rectangle"]["corners"],
                    "outer_corners_floor": [],
                    "order": "counter-clockwise from the south-west corner: SW, SE, NE, NW",
                    "inner_size_m": [round(closed["rectangle"]["width_m"], 3), round(closed["rectangle"]["depth_m"], 3)],
                    "checks": closed["checks"]},
        "walls": [keep(w) for w in walls],
        "not_included": list(ws.NOT_INCLUDED),
        # Every annotated door / door frame / window that is deliberately NOT an opening, with
        # the reason. A consumer that wants to place a leaf or a pane needs to know where they
        # are; this is the list it reads, and it is why `openings: []` on a wall means "no hole
        # here", not "nothing was looked at".
        "openings_not_taken": list(openings_skipped),
        "known_issues": [
            "only doorways with no door leaf are handed over as openings: a door leaf and a window "
            "are their own items (see not_included) and get no hole cut for them. `openings_not_taken` "
            "says which annotated instances were left out and why",
            "exterior wall thickness is assumed: the annotation has no outside face either",
            "key frames carry no reprojection_px: they use GT poses, so a reprojection error would "
            "measure nothing. `mask_agreement` replaces it - the share of the wall's own vertices "
            "that land inside the annotation's 2D mask for that wall",
            "openings carry source gt_instance, not ray_test: they are annotated instances, not holes the rays saw",
            "the outline's outer corners are left empty; consumers build the prism from the line and thickness",
            "key frames are picked from 2D instance masks: a wall hidden by a curtain has no mask there",
        ],
    }


def readme(doc, out, scene):
    rows = []
    for w in doc["walls"]:
        kfs = "、".join(f"{k['source_frame_id']}（{k['view_angle_deg']:.0f}°）" for k in w["key_frames"]) or "—"
        rows.append(f"| {w['id']} | {'外墙' if w['role'] == 'exterior' else '内墙'} "
                    f"{w['thickness_m'] * 100:.0f} cm（{w['thickness_kind']}） | {w['length_m']:.2f} m | "
                    f"{len(w['openings'])} | {kfs} |")
    text = README.format(scene=scene.name, run=doc["producer"]["run"], out=out, w=doc["camera"]["width"],
                         h=doc["camera"]["height"], storey=doc["storey"]["height_m"],
                         definition="docs/amb3r-floor-frame.md", scale_source="-", scale_value=0.0,
                         top=doc["storey"]["evidence"].get("ceiling_z_m", doc["storey"]["height_m"]),
                         te=doc["thickness_defaults"]["exterior_m"], ti=doc["thickness_defaults"]["interior_m"],
                         rows="\n".join(rows), issues="\n".join(f"- {x}" for x in doc["known_issues"])
                         )
    text = _replace_section(text, "## 坐标", COORD_ZH.format(floor_z=doc["frame"]["floor_z"]))
    text = text.replace(
        "5. `openings` 是射线检验**测到的洞**：从这个洞看过去，墙从下到上都被射线穿过，上方还有实墙封住（`closed_above`）。"
        "`from_m` / `to_m` 是沿墙位置（从 start 起算），`z_floor` 是离地高度范围（地面坐标系），"
        "`corners_floor` 是这个洞的四角（顺序同墙段：起点下、终点下、终点上、起点上）。生成墙体时要在这些位置留出洞。",
        "5. `openings` 只放**没有门的门洞**（`source: gt_instance`），不是射线测出来的："
        "标注里一个门框上没挂门扇，就按它在墙上占的范围写成一个洞。`from_m` / `to_m` 是沿墙位置（从 start 起算），"
        "`z_floor` 是离地高度范围（地面坐标系），`corners_floor` 是这个洞的四角（顺序同墙段：起点下、终点下、终点上、起点上）。"
        "生成墙体时要在这些位置留出洞。"
    ).replace(
        "   - 这里只有**开着的**门洞和通道。关着的门扇、玻璃窗会把射线挡在门面或玻璃面上，测不到，"
        "所以某个墙的 `openings` 为空，只说明“没测到洞”，不等于“这面墙没有门或窗”。",
        "   - **门扇和窗不是洞，不在这里。** 它们是各自的东西（见 `not_included`），墙不为它们开洞："
        "关着的门扇填满门框，窗是一块玻璃。哪些标注实例因此没被当作开口，逐条记在 `openings_not_taken` 里，"
        "带上原因。所以某个墙的 `openings` 为空，表示这面墙上没有可通行的洞，而不是没有门窗。"
    ).replace(
        "- `kind` 只写 `opening`，不写门或窗：射线只能说明那里是个洞，不能说明洞里是门、是窗还是什么都没有。",
        "- `kind` 统一写 `opening`，`instance` 记的是它来自哪个标注实例（例如 `doorframe#63`）。"
        "门扇、窗本身在 `not_included` 里。"
    ).replace(
        "   - `bounds` 说明洞的四条边各由什么界定：左、右、上为 `solid`（射线停在实墙上）；"
        "下边为 `floor`（洞和地面之间没有实墙，贴地那一条射线打不到，按到地面处理，`z_floor` 从 0 起，"
        "`seen_bottom_m` 是实际看到的最低处）或 `solid`（下方有墙，例如窗台）。只有四边都这样界定的洞才会写进来。",
        "   - `bounds` 在这里一律为左、右、上 `solid`、下边 `floor` 或 `solid`，由标注实例的底面高度决定"
        "（门按落地处理，窗的下边记 `solid`）。这是**从标注推出的**，不是射线测的，"
        "所以它只描述布局，不构成“这面墙其余部分都是实墙”的保证。")
    text = text.replace(
        "- 相机位姿是 AMB3R 自己的位姿（转到 `amb3r_floor`），不是 ScanNet 的 GT 位姿：墙是用 AMB3R 抬到三维的，"
        "用它自己的位姿投回去才对得上（实测 6–16 px，GT 位姿 24–79 px）。",
        "- 相机位姿是 **ScanNet 的 GT 位姿**，和墙在同一个 `scannet_scan` 坐标系里（z 同样减去了 `frame.floor_z`）。")
    return text


def _replace_section(text: str, head: str, body: str) -> str:
    """Swap the body of a `## ` section of the README for another."""
    start = text.index(head)
    end = text.index("\n## ", start + len(head))
    return text[:start] + body.rstrip("\n") + "\n" + text[end:]


COORD_ZH = """## 坐标

- 坐标系是 **`scannet_scan`**：ScanNet 的原始扫描坐标系，也就是标注网格 `<scene>_vh_clean_2.ply` 和 GT 相机位姿所在的坐标系（没有套 `axisAlignment`），米，+Z 向上。和旧的物体 GT 交接 `gt/` 是同一个坐标系。
- **高度从标注地面量起**：x、y 是扫描坐标系里的原值；z 减去了 `frame.floor_z`（{floor_z} m，标注 `floor` 实例顶点 z 的中位数），所以墙站在 z = 0 上、往上 `height_m`，和墙体交接的约定一致。要回到原始扫描坐标，把 z 加回 `frame.floor_z`。
- 二维点 `[x, y]` 是 z = 0 时的地面坐标，字段名带 `_floor`。
- **它和我们自己的墙体交接 `walls_harness/`（`amb3r_floor`）不在同一个坐标系**，不能直接叠放。要叠放时用 `frame.to_amb3r_floor.matrix`（本文件坐标 → amb3r_floor 的刚体变换，和 `gt_amb3r_floor/` 用的是同一个变换），只用于对照。
- 关键帧的 `camera_to_world` 也在这个坐标系里（同样减去了 `frame.floor_z`），OpenCV 约定（+X 右、+Y 下、+Z 前），内参在 `camera`。用法：`p_cam = inverse(camera_to_world) @ p`，`u = fx·x/z + cx`，`v = fy·y/z + cy`。
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = ap.parse_args()
    build(a.scene.resolve(), a.output)


if __name__ == "__main__":
    main()
