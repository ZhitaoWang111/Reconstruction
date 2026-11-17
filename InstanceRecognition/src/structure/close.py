"""Step 3e: exterior or interior, a closed outline, one storey height, wall thickness.

    python -m src.structure.close --run struct_v4_win5

Everything here is inferred on top of the observed walls, and every output says which part
is observed and which is inferred or assumed.

 1. Seen space. Along every ray (rays.py) the stretch from the camera to 10 cm short of the
    surface is empty, between 0.2 and 2.2 m high. Counted on a 5 cm floor grid.
 2. Exterior or interior. Behind an exterior wall is the outside, which no frame sees;
    behind an interior wall is another room, seen through a doorway. A wall whose
    0.15-0.8 m strip behind it is mostly seen space is interior.
 3. Main axis: wall facings weighted by area, averaged modulo 90 degrees. Walls within
    SNAP_DEG of an axis are turned onto it about their own middle.
 4. Rectangle. Each side sits on the exterior walls facing that way (median of their
    points); a side with no exterior wall sits at the edge of the seen space. Checked: how
    much seen space falls outside it, and how much of its inside was seen.
 5. Interior walls are extended from each end, along themselves, to the first wall line
    they meet within MAX_EXT - unless the rays see through the stretch they would cover.
 6. One storey height for every wall: STOREY, assumed. No frame looks at the ceiling.
 7. Thickness: THICK, assumed, laid behind the room-side face (the face stays where it was
    seen). What the data says about thickness is attached as evidence and never used: an
    interior wall's other face seen as a separate plane, and a door leaf set back behind
    the face (a lower bound).

Writes `closed.json` to artifacts/structure/<run>/.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import ndimage

from . import walls as wl
from .frames import RAYS, FloorFrame, Rays, run_dir

STOREY = 2.80                                   # m; assumed, not measured
THICK = {"exterior": 0.30, "interior": 0.15}    # m; assumed, not measured
CELL = 0.05            # m; seen-space grid
SEEN_MIN = 3           # ray samples in a cell to call it seen empty
BEHIND = (0.15, 0.80)  # m behind a wall face: the strip tested for seen space
INTERIOR = 0.5         # share of that strip seen -> interior
SNAP_DEG = 10.0
MAX_EXT = 1.5          # m; how far an interior wall may be extended at one end
PARALLEL_DEG = 20.0    # lines closer to parallel than this do not meet
SIDES = ("W", "S", "E", "N")                    # side k has inward normal axes[k]: +s, +t, -s, -t


def seen_space(P, C):
    lo = np.floor(np.minimum(P[:, :2].min(0), C[:, :2].min(0))) - 0.5
    n = np.ceil((np.maximum(P[:, :2].max(0), C[:, :2].max(0)) + 0.5 - lo) / CELL).astype(int)
    cnt = np.zeros(n, np.int32)
    L = np.linalg.norm(P - C, axis=1)
    for t in np.linspace(0.03, 0.97, 32):
        Q = C + t * (P - C)
        ok = (L * (1 - t) > 0.10) & (Q[:, 2] > 0.2) & (Q[:, 2] < 2.2)
        ij = np.floor((Q[ok, :2] - lo) / CELL).astype(int)
        np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    return cnt >= SEEN_MIN, lo


def cell_of(p, lo, shape):
    ij = np.floor((np.atleast_2d(p) - lo) / CELL).astype(int)
    return np.clip(ij, 0, np.array(shape) - 1)


def thickness_hints(r, walls, others, X_all, points, masks):
    """What the data says about one wall's thickness. Recorded, never used."""
    hints = []
    c, n, u = np.array(r["point"]), np.array(r["normal"]), np.array(r["along"])
    for o in walls + others:                       # a plane facing the other way just behind this one
        if o is r or (np.array(o["normal"]) @ n) > -np.cos(np.radians(20)):
            continue
        d = -((o["pts"] - c) @ n); t = (o["pts"] - c) @ u
        ov = (t > r["a"][0]) & (t < r["a"][1])
        if ov.mean() < 0.5 or not (0.02 < np.median(d[ov]) < 0.45):
            continue
        hints.append({"kind": "two_faces", "other_plane": o["id"], "frames": o["frames"],
                      "gap_cm": float(np.median(d[ov]) * 100),
                      "iqr_cm": (np.percentile(d[ov], [25, 75]) * 100).round(1).tolist(),
                      "angle_deg": float(np.degrees(np.arccos(np.clip(-(np.array(o["normal"]) @ n), -1, 1)))),
                      "note": f"other face seen in {len(o['frames'])} frame(s); one surface's depth drifts up to ~24 cm between frames"})
    rec = []
    for m in masks:                                # a door leaf behind the face: the wall is at least that thick
        if m["label"] != "door":
            continue
        P2 = X_all[points["mask"] == m["mask"], :2]
        t = (P2 - c) @ u; d = -((P2 - c) @ n)
        if ((t > r["a"][0]) & (t < r["a"][1])).mean() < 0.5 or not (0.0 < np.median(d) < 0.4):
            continue
        rec.append((m["mask"], m["frame"], float(np.median(d) * 100)))
    if rec:
        v = [x[2] for x in rec]
        hints.append({"kind": "door_recess", "masks": [x[0] for x in rec], "frames": [x[1] for x in rec],
                      "recess_cm": [round(min(v), 1), round(max(v), 1)], "median_cm": float(np.median(v)),
                      "note": "a door leaf set back behind the face: lower bound for the thickness"})
    return hints


def close(run: str, storey: float = STOREY) -> dict:
    d = run_dir(run)
    W = json.loads((d / "walls.json").read_text())
    g = np.load(d / "geom.npz")
    X = g["X"][g["wall_first"]].astype(np.float64); plane = g["wall_plane"]
    rays = Rays(FloorFrame(W["floor"]), RAYS)
    walls = [dict(r) for r in W["walls"]]
    others = [dict(r) for r in W["rejected"]]      # the far side of a partition is often only here
    for i, r in enumerate(walls):
        r["pts"] = X[plane == i, :2]
    for i, r in enumerate(others):
        r["pts"] = X[plane == len(walls) + i, :2]

    # ---- 1. seen space
    seen, lo = seen_space(rays.P, rays.C)
    print(f"seen space: {seen.sum() * CELL ** 2:.1f} m2")

    # ---- 2. exterior / interior
    for r in walls:
        c, n, u = np.array(r["point"]), np.array(r["normal"]), np.array(r["along"])
        A, Dd = np.meshgrid(np.linspace(r["a"][0] + 0.1, r["a"][1] - 0.1, 40), -np.linspace(*BEHIND, 14))
        ij = cell_of(c + A.reshape(-1, 1) * u + Dd.reshape(-1, 1) * n, lo, seen.shape)
        r["behind_seen"] = float(seen[ij[:, 0], ij[:, 1]].mean())
        r["role"] = "interior" if r["behind_seen"] > INTERIOR else "exterior"

    # ---- 3. main axis and snapping
    ang = np.radians([r["heading_deg"] for r in walls]); wgt = np.array([r["area_m2"] for r in walls])
    theta = float(np.angle((wgt * np.exp(4j * ang)).sum()) / 4)
    axes = [np.array([np.cos(theta + k * np.pi / 2), np.sin(theta + k * np.pi / 2)]) for k in range(4)]
    for r in walls:
        n0 = np.array(r["normal"])
        k = int(np.argmax([n0 @ e for e in axes]))
        dev = float(np.degrees(np.arctan2(axes[k][0] * n0[1] - axes[k][1] * n0[0], axes[k] @ n0)))
        r["axis"] = k if abs(dev) <= SNAP_DEG else None
        r["snap_deg"] = -dev if r["axis"] is not None else 0.0
        n1 = axes[k] if r["axis"] is not None else n0
        u1 = np.array([-n1[1], n1[0]])
        mid = np.median(r["pts"], 0)
        al = (r["pts"] - mid) @ u1
        r["line"] = {"c": mid, "n": n1, "u": u1, "a": [float(np.percentile(al, 0.5)), float(np.percentile(al, 99.5))]}

    # ---- 4. rectangle
    ex, ey = axes[0], axes[1]
    fi, fj = np.nonzero(seen); F = lo + (np.stack([fi, fj], 1) + 0.5) * CELL
    sides = []
    for k in range(4):
        on = [r for r in walls if r["role"] == "exterior" and r["axis"] == k]
        pos = (float(np.median(np.concatenate([r["pts"] @ axes[k] for r in on]))) if on
               else float(np.percentile(F @ axes[k], 0.5)))
        sides.append({"k": k, "side": SIDES[k], "normal": axes[k].tolist(), "offset": pos,
                      "from": "observed" if on else "seen_space", "walls": [r["id"] for r in on]})
    s0, t0 = sides[0]["offset"], sides[1]["offset"]
    s1, t1 = -sides[2]["offset"], -sides[3]["offset"]
    corners = [s0 * ex + t0 * ey, s1 * ex + t0 * ey, s1 * ex + t1 * ey, s0 * ex + t1 * ey]   # SW SE NE NW
    seg_of = {0: (corners[3], corners[0]), 1: (corners[0], corners[1]), 2: (corners[1], corners[2]), 3: (corners[2], corners[3])}
    for sd in sides:                                # each side runs with the room on its left
        p0, p1 = seg_of[sd["k"]]; dv = p1 - p0; L = float(np.linalg.norm(dv)); dv = dv / L
        cover = []
        for r in walls:
            if r["id"] in sd["walls"]:
                t = np.clip((r["pts"] - p0) @ dv, 0, L)
                cover.append((float(np.percentile(t, 0.5)), float(np.percentile(t, 99.5))))
        cover.sort(); merged = []
        for c in cover:
            if merged and c[0] <= merged[-1][1] + 0.05:
                merged[-1] = (merged[-1][0], max(merged[-1][1], c[1]))
            else:
                merged.append(c)
        sd.update({"p0": p0.tolist(), "p1": p1.tolist(), "length_m": L, "observed": merged,
                   "observed_m": float(sum(b - a for a, b in merged))})
    ins = (F @ ex >= s0 - CELL) & (F @ ex <= s1 + CELL) & (F @ ey >= t0 - CELL) & (F @ ey <= t1 + CELL)
    I, J = np.meshgrid(np.arange(seen.shape[0]), np.arange(seen.shape[1]), indexing="ij")
    G = lo + (np.stack([I, J], -1) + 0.5) * CELL
    inside = (G @ ex >= s0) & (G @ ex <= s1) & (G @ ey >= t0) & (G @ ey <= t1)
    unseen, k_ = ndimage.label(inside & ~seen)
    big = max([(unseen == k).sum() for k in range(1, k_ + 1)] or [0]) * CELL ** 2
    rect = {"corners": [c.tolist() for c in corners], "width_m": s1 - s0, "depth_m": t1 - t0, "area_m2": (s1 - s0) * (t1 - t0),
            "axis_deg": float(np.degrees(theta)), "seen_outside_m2": float((~ins).sum() * CELL ** 2),
            "seen_outside_share": float((~ins).mean()), "inside_seen_share": float((inside & seen).sum() / inside.sum()),
            "largest_unseen_inside_m2": float(big)}

    # ---- 5. interior walls: extend each end to the first wall line it meets
    lines = [{"id": f"side{sd['k']}", "c": np.array(sd["p0"]), "n": np.array(sd["normal"]),
              "u": (np.array(sd["p1"]) - np.array(sd["p0"])) / sd["length_m"], "a": [0.0, sd["length_m"]]} for sd in sides]
    lines += [{"id": r["id"], **r["line"]} for r in walls if r["role"] == "interior"]
    for r in walls:
        L0 = r["line"]; r["ends"] = []
        if r["role"] != "interior":
            continue
        for sgn, a_end in ((-1, L0["a"][0]), (1, L0["a"][1])):
            p = L0["c"] + a_end * L0["u"]; dv = sgn * L0["u"]
            best = None
            for M in lines:
                if M["id"] == r["id"]:
                    continue
                den = dv @ M["n"]
                if abs(den) < np.sin(np.radians(PARALLEL_DEG)):
                    continue
                t = float((M["c"] - p) @ M["n"] / den)
                if not (-0.3 <= t <= MAX_EXT):
                    continue
                h = p + t * dv; ah = float((h - M["c"]) @ M["u"])
                slack = 0.05 if M["id"].startswith("side") else MAX_EXT
                if not (M["a"][0] - slack <= ah <= M["a"][1] + slack):
                    continue
                if best is None or t < best[0]:
                    best = (t, M["id"], h)
            end = {"side": "start" if sgn < 0 else "end", "from": p.tolist()}
            if best is None:
                end.update({"to": p.tolist(), "ext_m": 0.0, "meets": None, "why": f"no wall line within {MAX_EXT} m"})
            else:
                t, other, h = best
                end.update({"to": h.tolist(), "ext_m": t, "meets": other})
                if t > CELL:
                    a0_, a1_ = sorted([a_end, a_end + sgn * t])
                    solid, through = wl.ray_grid(rays, L0["c"], L0["n"], a0_, a1_)
                    free = (through >= wl.MIN_RAYS) & (through > 3 * solid)
                    open_share = float(((free.sum(1) >= 3) & ((solid >= wl.MIN_RAYS).sum(1) == 0)).mean())
                    end["open_share"] = open_share
                    if open_share > 0.5:
                        end.update({"to": p.tolist(), "ext_m": 0.0, "why": f"rays pass through {open_share:.0%} of the extension"})
            r["ends"].append(end)

    # ---- 6. final walls: observed and inferred pieces
    top = max(r["z"][1] for r in walls)
    final = []
    for sd in sides:
        p0, dv = np.array(sd["p0"]), (np.array(sd["p1"]) - np.array(sd["p0"])) / sd["length_m"]
        marks = [0.0] + [x for ab in sd["observed"] for x in ab] + [sd["length_m"]]
        pieces = [{"p0": (p0 + marks[i] * dv).tolist(), "p1": (p0 + marks[i + 1] * dv).tolist(),
                   "kind": "observed" if i % 2 == 1 else "inferred"}
                  for i in range(len(marks) - 1) if marks[i + 1] - marks[i] >= 0.01]
        final.append({"id": f"ext_{sd['side']}", "side": sd["k"], "role": "exterior", "walls": sd["walls"],
                      "normal": sd["normal"], "pieces": pieces, "position_from": sd["from"]})
    for r in walls:
        if r["role"] != "interior":
            continue
        L0 = r["line"]
        a0 = L0["c"] + L0["a"][0] * L0["u"]; a1 = L0["c"] + L0["a"][1] * L0["u"]
        e0, e1 = r["ends"]
        pieces = []
        if e0["ext_m"] > 0.01:
            pieces.append({"p0": e0["to"], "p1": a0.tolist(), "kind": "inferred"})
        if e0["ext_m"] < -0.01:
            a0 = np.array(e0["to"])
        if e1["ext_m"] < -0.01:
            a1 = np.array(e1["to"])
        pieces.append({"p0": a0.tolist(), "p1": a1.tolist(), "kind": "observed"})
        if e1["ext_m"] > 0.01:
            pieces.append({"p0": a1.tolist(), "p1": e1["to"], "kind": "inferred"})
        final.append({"id": r["id"], "role": "interior", "walls": [r["id"]], "normal": L0["n"].tolist(), "pieces": pieces})
    for f in final:
        for p in f["pieces"]:
            p["length_m"] = float(np.linalg.norm(np.array(p["p1"]) - np.array(p["p0"])))
        f["length_m"] = sum(p["length_m"] for p in f["pieces"])
        f["inferred_m"] = sum(p["length_m"] for p in f["pieces"] if p["kind"] == "inferred")

    # ---- 7. thickness
    points = np.load(d / "points.npz")
    masks = json.loads((d / "masks.json").read_text())
    X_all = g["X"].astype(np.float64)
    for r in walls:
        r["thickness_hint"] = thickness_hints(r, walls, others, X_all, points, masks)
    for f in final:
        t = THICK[f["role"]]
        f["thickness_m"] = t; f["thickness_kind"] = "assumed"
        f["normal_to_back"] = (-np.array(f["normal"])).tolist()
        f["thickness_hint"] = [h for w_ in walls if w_["id"] in f["walls"] for h in w_["thickness_hint"]]

    # ---- report
    print(f"main axis {np.degrees(theta):+.2f} deg")
    for r in walls:
        print(f"  {r['id']} behind seen {r['behind_seen']:4.0%} -> {r['role']}; "
              + (f"snapped to {SIDES[r['axis']]} by {r['snap_deg']:+.1f} deg" if r["axis"] is not None else "not snapped"))
    print(f"rectangle {rect['width_m']:.2f} x {rect['depth_m']:.2f} m; seen space outside {rect['seen_outside_share']:.1%}, "
          f"inside seen {rect['inside_seen_share']:.0%}, largest unseen {rect['largest_unseen_inside_m2']:.2f} m2")
    for f in final:
        print(f"  {f['id']:6} {f['role']:8} {f['length_m']:.2f} m, inferred {f['inferred_m']:.2f} m, thickness {f['thickness_m']:.2f} m")
    print(f"storey {storey:.2f} m (assumed); highest wall point seen {top:.2f} m")

    for r in walls:
        r["line"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r["line"].items()}
    strip = lambda r: {k: v for k, v in r.items() if k != "pts"}
    doc = {"run": run, "note": "inferred on top of the observed walls; pieces say observed or inferred",
           "storey": {"height_m": storey, "kind": "assumed",
                      "why": "no frame looks at the ceiling: every point above 2.5 m comes from the top 10% of image rows",
                      "observed_wall_top_m": float(top)},
           "params": {"cell_m": CELL, "seen_min": SEEN_MIN, "behind_m": list(BEHIND), "interior_share": INTERIOR,
                      "snap_deg": SNAP_DEG, "max_ext_m": MAX_EXT, "parallel_deg": PARALLEL_DEG},
           "thickness": {"exterior_m": THICK["exterior"], "interior_m": THICK["interior"], "kind": "assumed",
                         "applied": "behind the room-side face; the face stays where it was seen"},
           "axis_deg": float(np.degrees(theta)), "rectangle": rect, "sides": sides,
           "walls": [strip(r) for r in walls], "final": final,
           "seen": {"lo": lo.tolist(), "cell_m": CELL, "shape": list(seen.shape),
                    "cells": np.flatnonzero(seen.reshape(-1)).tolist()}}
    (d / "closed.json").write_text(json.dumps(doc, ensure_ascii=False))
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--storey", type=float, default=STOREY)
    a = ap.parse_args()
    close(a.run, a.storey)


if __name__ == "__main__":
    main()
