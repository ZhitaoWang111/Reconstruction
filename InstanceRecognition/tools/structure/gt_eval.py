"""Structure recognition (src.structure) against the ScanNet GT wall surface: one fixed set of numbers per run.

    .venv/bin/python -m tools.structure.gt_eval --run struct_v4_win5                 # print, write gt_eval.json
    .venv/bin/python -m tools.structure.gt_eval --run struct_v4_win5 --save-baseline # also keep it as the baseline
    .venv/bin/python -m tools.structure.gt_eval --run struct_v4_win5 --compare       # this run vs the baseline

Evaluation only: nothing in src/ reads it. The reference is the space the GT walls occupy:
the faces of the annotated mesh labelled `wall`, sampled uniformly (2000 points per m2, seed 0)
and moved into amb3r_floor through the oracle Sim(3). Points per area, so a share of points is
a share of wall surface. The 2D instance annotation is not used (too coarse to judge masks by).

What is measured, against what was handed off (the wall solids: footprint x [0, height_m],
minus the openings):

  completeness  share of GT wall surface inside a handed-off wall solid (within TOL), and the
                stricter share within TOL of a room-side face (a 30 cm solid swallows a
                15 cm offset; the face does not); per GT face.
  offset        per GT face, where the GT wall surface sits relative to the room-side face of
                the wall it falls on (+ = GT is behind our face, i.e. our face is into the room).
  support       our room-side faces sampled every 5 cm: share with GT wall surface within
                SUPPORT; of the rest, share above the GT wall top, share in front of some other
                GT surface (curtain, cabinet, door, window), share with no GT surface at all.
  height        per wall, the GT wall top (p99 of GT wall points on it) against height_m.
  holes         GT wall surface inside the handed-off openings (a real hole has none), and the
                GT doors / windows / doorway (from the mesh annotation) found or missed.
  chain         share of GT wall surface reached by each stage: a ray endpoint, a wall-mask
                point, a point in a kept plane, a handed-off wall - where the evidence is lost.
  room          the closed rectangle per axis against the GT faces its opposite sides lie on.
  thickness     the thickness hints against the GT measurements.

Results go to artifacts/structure/<run>/gt_eval.json; the baseline to
artifacts/structure/gt_eval_baseline.json. `--compare` prints every headline number that moved.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src import config
from src.geometry.amb3r_mask_source import DEFAULT_SIM3
from src.geometry.floor_frame import Frame, scannet_scan_to_frame
from src.structure.frames import OUT_ROOT, RAYS, axis_alignment_for_eval

SCENE = config.SCANNET_ROOT / "scene0000_00"
GT_DATA = config.ROOT / "docs/assets/structure-gt/data.js"
HANDOFF = config.ARTIFACTS_ROOT / "handoff/scene0000_00/walls_harness"
BASELINE = OUT_ROOT / "gt_eval_baseline.json"
GT_SAMPLES_PER_M2 = 2000
TOL = 0.10            # m; a GT wall point this close to a handed-off solid counts as covered
SUPPORT = 0.15        # m; our face sample with GT wall surface this close (along its normal) is supported
FACE_STEP = 0.05      # m; sampling of our room-side faces
CELL = 0.10
BAND = (0.1, 2.3)
NEAR = 0.40          # m; GT wall points this close to one of our planes / sides are compared with it
OPENING_LABELS = {"door": "door", "window": "window", "opening": "doorway"}


# ---------------------------------------------------------------- GT in amb3r_floor
def gt_to_frame() -> np.ndarray:
    """ScanNet axis-aligned frame (the GT page's) -> amb3r_floor."""
    S = scannet_scan_to_frame(Frame.load(), json.loads(DEFAULT_SIM3.read_text())["sim3_oracle_scale"])
    return S @ np.linalg.inv(axis_alignment_for_eval())


def gt_structure() -> dict:
    s = GT_DATA.read_text()
    return json.loads(s[s.index("{"):s.index(";\nwindow.STRUCT.pos")])


def gt_gaps() -> list:
    """The thicknesses the GT page measured on the annotated mesh, with where (p0 -> p1) in amb3r_floor."""
    if not GT_DATA.is_file():
        return []
    G = gt_to_frame()
    T = lambda p: (G[:3, :3] @ np.asarray(p, dtype=np.float64) + G[:3, 3]).tolist()
    return [{"name": g["name"], "cm": g["cm"], "spread_cm": g["spread_cm"], "door": g["b"].startswith("door#"),
             "p0": T(g["p0"]), "p1": T(g["p1"])} for g in gt_structure().get("gaps", [])]


def gt_items() -> list:
    """GT doors, windows and the inferred bathroom doorway, as 3D corners in amb3r_floor."""
    if not GT_DATA.is_file():
        return []
    G = gt_to_frame()
    out = []
    for it in gt_structure()["items"]:
        kind = OPENING_LABELS.get(it["label"])
        if kind is None:
            continue
        C = np.array([G[:3, :3] @ np.asarray(q) + G[:3, 3] for q in it["corners"]])
        out.append({"key": it["key"], "name": it["name"], "kind": kind, "corners": C.tolist()})
    return out


@lru_cache(maxsize=1)
def gt_surface() -> dict:
    """The annotated mesh sampled by area, in amb3r_floor: points, their GT label and instance, and the wall subset."""
    from pagegen import build_structure_gt as gsg
    P, _, Fc, owner, labels, _ = gsg.load(SCENE)
    pts, _, fo, _, _ = gsg.sample_faces(P, Fc, owner, np.random.default_rng(0), per_m2=GT_SAMPLES_PER_M2)
    S = scannet_scan_to_frame(Frame.load(), json.loads(DEFAULT_SIM3.read_text())["sim3_oracle_scale"])
    pts = pts @ S[:3, :3].T + S[:3, 3]
    lab = np.array([labels.get(int(o), "unannotated") for o in fo])
    return {"points": pts, "label": lab, "instance": fo, "wall": lab == "wall", "per_m2": GT_SAMPLES_PER_M2}


class Solids:
    """The handed-off walls as solids: footprint (room face + thickness behind it) x [0, height_m], minus openings."""

    def __init__(self, H: dict):
        self.walls = []
        for w in H["walls"]:
            a, b = np.asarray(w["start_floor"]), np.asarray(w["end_floor"])
            L = float(np.linalg.norm(b - a)); d = (b - a) / L
            self.walls.append({"id": w["id"], "a": a, "d": d, "back": np.asarray(w["normal_to_back"]), "L": L,
                               "t": float(w["thickness_m"]), "h": float(w["height_m"]),
                               "fp": np.asarray(w["footprint_floor"]), "openings": w["openings"]})

    def distance(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per point: distance to the nearest solid (0 inside), which wall, and whether it lies in that wall's opening."""
        best = np.full(len(X), np.inf); which = np.full(len(X), -1); in_hole = np.zeros(len(X), bool)
        for k, w in enumerate(self.walls):
            dxy = polygon_distance(X[:, :2], w["fp"])
            dz = np.maximum(np.maximum(-X[:, 2], X[:, 2] - w["h"]), 0)
            dist = np.hypot(dxy, dz)
            s = (X[:, :2] - w["a"]) @ w["d"]; q = (X[:, :2] - w["a"]) @ w["back"]
            hole = np.zeros(len(X), bool)
            for o in w["openings"]:
                hole |= (s > o["from_m"]) & (s < o["to_m"]) & (X[:, 2] > o["z_floor"][0]) & (X[:, 2] < o["z_floor"][1]) \
                    & (q > -TOL) & (q < w["t"] + TOL)
            closer = dist < best
            best[closer], which[closer], in_hole[closer] = dist[closer], k, hole[closer]
        return best, which, in_hole

    def face_distance(self, X: np.ndarray) -> np.ndarray:
        """Per point: distance to the nearest room-side face (the sheet start->end x [0, height_m]), openings left open."""
        best = np.full(len(X), np.inf)
        for w in self.walls:
            s = (X[:, :2] - w["a"]) @ w["d"]; q = (X[:, :2] - w["a"]) @ w["back"]
            ds = np.maximum(np.maximum(-s, s - w["L"]), 0)
            dz = np.maximum(np.maximum(-X[:, 2], X[:, 2] - w["h"]), 0)
            d = np.sqrt(ds ** 2 + q ** 2 + dz ** 2)
            for o in w["openings"]:
                d[(s > o["from_m"]) & (s < o["to_m"]) & (X[:, 2] > o["z_floor"][0]) & (X[:, 2] < o["z_floor"][1])] = np.inf
            best = np.minimum(best, d)
        return best

    def face_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Our room-side faces every FACE_STEP along x height, openings left out: (points, wall index)."""
        pts, idx = [], []
        for k, w in enumerate(self.walls):
            s = np.arange(FACE_STEP / 2, w["L"], FACE_STEP); z = np.arange(FACE_STEP / 2, w["h"], FACE_STEP)
            S, Z = np.meshgrid(s, z, indexing="ij"); S, Z = S.ravel(), Z.ravel()
            keep = np.ones(len(S), bool)
            for o in w["openings"]:
                keep &= ~((S > o["from_m"]) & (S < o["to_m"]) & (Z > o["z_floor"][0]) & (Z < o["z_floor"][1]))
            xy = w["a"] + S[keep, None] * w["d"]
            pts.append(np.c_[xy, Z[keep]]); idx.append(np.full(int(keep.sum()), k))
        return np.concatenate(pts), np.concatenate(idx)


def polygon_distance(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from 2D points to a convex polygon (0 inside)."""
    n = len(poly)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    sign = 1.0 if area > 0 else -1.0
    inside = np.ones(len(p), bool); dmin = np.full(len(p), np.inf)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]; e = b - a
        cross = e[0] * (p[:, 1] - a[1]) - e[1] * (p[:, 0] - a[0])
        inside &= sign * cross >= 0
        t = np.clip(((p - a) @ e) / max(e @ e, 1e-12), 0, 1)
        dmin = np.minimum(dmin, np.linalg.norm(p - (a + t[:, None] * e), axis=1))
    return np.where(inside, 0.0, dmin)


# ---------------------------------------------------------------- 3D
def eval_occupancy(run: str, H: dict) -> dict:
    """GT wall surface against the handed-off solids, and the evidence chain that led to them."""
    gs = gt_surface()
    G = gs["points"][gs["wall"]]
    per_m2 = gs["per_m2"]
    solids = Solids(H)
    dist, which, in_hole = solids.distance(G)
    covered = (dist < TOL) & ~in_hole
    on_face = solids.face_distance(G) < TOL
    # signed position of the GT surface relative to the room face of its nearest wall (+ = behind it)
    q = np.full(len(G), np.nan)
    for k, w in enumerate(solids.walls):
        m = which == k
        q[m] = (G[m, :2] - w["a"]) @ w["back"]
    # evidence chain: GT wall points reached by each stage (within CELL)
    d = OUT_ROOT / run
    g = np.load(d / "geom.npz")
    Xw = g["X"][g["wall_first"]].astype(np.float64); plane = g["wall_plane"]
    nk = len(json.loads((d / "walls.json").read_text())["walls"])
    P = np.load(RAYS)["P"].astype(np.float64)
    reach = lambda Q: cKDTree(Q).query(G, distance_upper_bound=CELL)[0] < CELL if len(Q) else np.zeros(len(G), bool)
    chain = {"rays": reach(P), "masks": reach(Xw), "planes": reach(Xw[(plane >= 0) & (plane < nk)]), "on_face": on_face}
    # per handed-off wall: the GT wall points nearest to it (within NEAR of its solid)
    per_wall = []
    for k, w in enumerate(solids.walls):
        m = (which == k) & (dist < NEAR)
        on = m & covered
        top = float(np.percentile(G[on, 2], 99)) if on.sum() > 50 else None
        per_wall.append({"wall": w["id"], "gt_m2": round(m.sum() / per_m2, 2),
                         "on_face": round(float((m & on_face).sum() / m.sum()), 4) if m.any() else None,
                         "offset_cm": round(float(np.median(q[on]) * 100), 1) if on.sum() > 50 else None,
                         "height_m": w["h"], "gt_top_m": round(top, 3) if top else None,
                         "height_minus_gt_cm": round((w["h"] - top) * 100, 1) if top else None,
                         **{f"chain_{c}": round(float((v & m).sum() / m.sum()), 4) if m.any() else None for c, v in chain.items()}})
    far = dist >= NEAR
    # support: our faces, sampled, against the GT surface
    F, fk = solids.face_samples()
    dw, _ = cKDTree(G).query(F, distance_upper_bound=SUPPORT)
    supported = dw < SUPPORT
    da, ia = cKDTree(gs["points"]).query(F[~supported], distance_upper_bound=0.5)
    near = np.where(da < 0.5, gs["label"][np.minimum(ia, len(gs["label"]) - 1)], "nothing within 0.5 m")
    why = Counter(near.tolist())
    for k, row in enumerate(per_wall):
        row["supported"] = round(float(supported[fk == k].mean()), 4) if (fk == k).any() else None
    return {"gt_wall_m2": round(len(G) / per_m2, 2), "tol_m": TOL, "near_m": NEAR,
            "completeness": round(float(covered.mean()), 4), "on_face": round(float(on_face.mean()), 4),
            "in_holes_m2": round(float(in_hole.sum() / per_m2), 3),
            "far_from_any_wall_m2": round(float(far.sum() / per_m2), 2),
            "chain": {c: round(float(v.mean()), 4) for c, v in chain.items()},
            "median_offset_cm": round(float(np.nanmedian(q[covered]) * 100), 1),
            "walls": per_wall,
            "support": {"step_m": FACE_STEP, "radius_m": SUPPORT, "samples": int(len(F)),
                        "supported": round(float(supported.mean()), 4), "handed_off_m2": round(len(F) * FACE_STEP ** 2, 2),
                        "unsupported_nearest_gt": {c: round(v / len(F), 4) for c, v in why.most_common()}}}


def eval_planes(W: dict) -> dict:
    """Each wall plane against the GT wall points within NEAR of it, inside its extent and height."""
    gs = gt_surface()
    G = gs["points"][gs["wall"]]
    rows = []
    for r in W["walls"] + W["rejected"]:
        c, n, u = (np.asarray(r[k]) for k in ("point", "normal", "along"))
        dd = (G[:, :2] - c) @ n; a = (G[:, :2] - c) @ u
        m = (np.abs(dd) < NEAR) & (a > r["a"][0]) & (a < r["a"][1]) & (G[:, 2] > r["z"][0]) & (G[:, 2] < r["z"][1])
        # n points into the room, so GT behind the plane has dd < 0: offset + = our plane sits into the room
        rows.append({"plane": r["id"], "kept": r["id"].startswith("W"), "gt_m2": round(m.sum() / gs["per_m2"], 2),
                     "offset_cm": round(float(-np.median(dd[m]) * 100), 1) if m.sum() > 50 else None})
    kept = [x for x in rows if x["kept"]]
    offs = [abs(x["offset_cm"]) for x in kept if x["offset_cm"] is not None]
    return {"planes": rows, "kept_without_gt": [x["plane"] for x in kept if x["offset_cm"] is None],
            "mean_abs_offset_cm": round(float(np.mean(offs)), 2) if offs else None}


def eval_room(C: dict) -> dict:
    """Each side of the closed rectangle against the GT wall points along it (middle 80% of the side, within NEAR)."""
    gs = gt_surface()
    G = gs["points"][gs["wall"]][:, :2]
    sides = []
    for sd in C["sides"]:
        p0, p1, n = (np.asarray(sd[k]) for k in ("p0", "p1", "normal"))
        L = float(np.linalg.norm(p1 - p0)); d = (p1 - p0) / L
        a = (G - p0) @ d; q = (G - p0) @ n
        m = (a > 0.1 * L) & (a < 0.9 * L) & (np.abs(q) < NEAR)
        # n points into the room: GT outside our side has q < 0, so + = our side sits inside the GT wall
        sides.append({"side": sd["side"], "from": sd["from"], "gt_m2": round(m.sum() / gs["per_m2"], 2),
                      "inward_cm": round(float(-np.median(q[m]) * 100), 1) if m.sum() > 50 else None})
    by = {x["side"]: x for x in sides}
    size = []
    for a, b, ours in (("W", "E", C["rectangle"]["width_m"]), ("S", "N", C["rectangle"]["depth_m"])):
        if by[a]["inward_cm"] is not None and by[b]["inward_cm"] is not None:
            gt = ours + (by[a]["inward_cm"] + by[b]["inward_cm"]) / 100
            size.append({"axis": f"{a}-{b}", "ours_m": round(ours, 4), "gt_m": round(gt, 4), "diff_cm": round((ours - gt) * 100, 2)})
    return {"sides": sides, "size": size, "area_m2": round(C["rectangle"]["area_m2"], 3),
            "seen_outside_share": round(C["rectangle"]["seen_outside_share"], 4)}


def eval_openings(H: dict) -> dict:
    """GT doors / windows / doorway placed on the handed-off wall they sit in, against the handed-off openings."""
    walls = Solids(H).walls
    gt = []
    for it in gt_items():
        C = np.asarray(it["corners"]); ctr = C.mean(0)
        best = None
        for w in walls:
            s_ = (ctr[:2] - w["a"]) @ w["d"]; q = (ctr[:2] - w["a"]) @ w["back"]
            if -0.2 < s_ < w["L"] + 0.2 and -NEAR < q < w["t"] + NEAR and (best is None or abs(q) < best[0]):
                best = (abs(q), w)
        if best is None:
            gt.append({**it, "wall": None}); continue
        w = best[1]; sa = (C[:, :2] - w["a"]) @ w["d"]
        gt.append({"key": it["key"], "name": it["name"], "kind": it["kind"], "wall": w["id"],
                   "a": [float(sa.min()), float(sa.max())], "z": [float(C[:, 2].min()), float(C[:, 2].max())]})
    # a door leaf and its frame are one opening: group GT items on one wall that overlap along it
    groups = []
    for g in sorted([g for g in gt if g["wall"]], key=lambda g: (g["wall"], g["a"][0])):
        if groups and groups[-1]["wall"] == g["wall"] and g["a"][0] < groups[-1]["a"][1]:
            gr = groups[-1]; gr["keys"].append(g["key"]); gr["a"][1] = max(gr["a"][1], g["a"][1])
            gr["z"] = [min(gr["z"][0], g["z"][0]), max(gr["z"][1], g["z"][1])]
        else:
            groups.append({"wall": g["wall"], "a": list(g["a"]), "z": list(g["z"]), "keys": [g["key"]], "name": g["name"], "kind": g["kind"]})
    rows = []
    for w in H["walls"]:
        for o in w["openings"]:
            a = [o["from_m"], o["to_m"]]
            row = {"wall": w["id"], "from_m": a[0], "to_m": a[1], "z": o["z_floor"], "width_m": o["width_m"],
                   "height_m": o["height_m"], "gt": None}
            best = max(((min(a[1], g["a"][1]) - max(a[0], g["a"][0]), g) for g in groups if g["wall"] == w["id"]),
                       key=lambda x: x[0], default=(0, None))
            if best[0] > 0:
                g = best[1]; g["found"] = True
                row["gt"] = {"name": g["name"], "kind": g["kind"],
                             "centre_cm": round(((a[0] + a[1]) - (g["a"][0] + g["a"][1])) / 2 * 100, 1),
                             "width_cm": round((a[1] - a[0] - (g["a"][1] - g["a"][0])) * 100, 1),
                             "bottom_cm": round((o["z_floor"][0] - g["z"][0]) * 100, 1),
                             "top_cm": round((o["z_floor"][1] - g["z"][1]) * 100, 1)}
            rows.append(row)
    for g in groups:
        g["found"] = bool(g.get("found"))
    return {"handed_off": rows, "gt": groups, "gt_not_on_a_wall": [g["name"] for g in gt if not g["wall"]],
            "gt_found": sum(g["found"] for g in groups), "gt_total": len(groups),
            "false": sum(1 for r in rows if r["gt"] is None)}


def eval_thickness(C: dict) -> dict:
    """Each thickness hint against the GT measurement taken on the same plane (by where the GT measured it)."""
    gaps = gt_gaps()
    hints = []
    for r in C["walls"]:
        c, n, u = (np.asarray(r[k]) for k in ("point", "normal", "along"))
        for h in r.get("thickness_hint", []):
            v = h.get("gap_cm", h.get("median_cm"))
            best = None
            for g in gaps:
                p = np.asarray(g["p0"][:2])
                if g["door"] != (h["kind"] == "door_recess"):
                    continue
                dn, da = abs((p - c) @ n), (p - c) @ u
                if dn < 0.3 and r["a"][0] - 0.2 < da < r["a"][1] + 0.2 and (best is None or dn < best[0]):
                    best = (dn, g)
            gt = best[1]["cm"] if best else None
            hints.append({"plane": r["id"], "kind": h["kind"], "cm": round(v, 1), "gt_cm": gt, "gt_name": best[1]["name"] if best else None,
                          "diff_cm": round(v - gt, 1) if gt is not None else None})
    return {"hints": hints, "assumed_m": C["thickness"], "gt_measured_cm": [g["cm"] for g in gaps]}


# ---------------------------------------------------------------- run
def evaluate(run: str) -> dict:
    d = OUT_ROOT / run
    W = json.loads((d / "walls.json").read_text())
    C = json.loads((d / "closed.json").read_text())
    H = json.loads((HANDOFF / "walls.json").read_text())
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    except OSError:
        commit = None
    doc = {"run": run, "commit": commit, "frame": Frame.load().doc["scale"],
           "occupancy": eval_occupancy(run, H), "planes": eval_planes(W), "room": eval_room(C),
           "openings": eval_openings(H), "thickness": eval_thickness(C)}
    doc["headline"] = headline(doc)
    return doc


def headline(doc: dict) -> dict:
    """The few numbers to compare between runs; every one is in the report."""
    o, p, r, op = doc["occupancy"], doc["planes"], doc["room"], doc["openings"]
    hs = [abs(w["height_minus_gt_cm"]) for w in o["walls"] if w["height_minus_gt_cm"] is not None]
    out = {"wall.on_face": o["on_face"], "wall.in_solid": o["completeness"], "wall.support": o["support"]["supported"],
           "wall.median_offset_cm": o["median_offset_cm"], "wall.gt_in_holes_m2": o["in_holes_m2"],
           "wall.gt_far_from_any_wall_m2": o["far_from_any_wall_m2"],
           "wall.mean_abs_height_error_cm": round(float(np.mean(hs)), 1) if hs else None,
           **{f"chain.{k}": v for k, v in o["chain"].items()},
           **{f"walls.{w['wall']}.on_face": w["on_face"] for w in o["walls"]},
           "planes.mean_abs_offset_cm": p["mean_abs_offset_cm"], "planes.kept_without_gt": len(p["kept_without_gt"]),
           "openings.gt_found": op["gt_found"], "openings.gt_total": op["gt_total"], "openings.false": op["false"],
           "room.area_m2": r["area_m2"]}
    for x in r["size"]:
        out[f"room.size_{x['axis']}_diff_cm"] = x["diff_cm"]
    return out


# ---------------------------------------------------------------- report
def report(doc: dict) -> None:
    o, p, r, op, t = (doc[k] for k in ("occupancy", "planes", "room", "openings", "thickness"))
    print(f"== {doc['run']}  (commit {doc['commit']}, scale {doc['frame']['source']} {doc['frame']['value']:.4f})")
    print(f"-- GT wall surface {o['gt_wall_m2']} m2: {o['on_face']:.1%} within {o['tol_m']} m of a handed-off room-side face, "
          f"{o['completeness']:.1%} inside a wall solid, {o['far_from_any_wall_m2']} m2 more than {o['near_m']} m from any wall, "
          f"{o['in_holes_m2']} m2 inside an opening; median offset {o['median_offset_cm']:+.1f} cm (+ = our face into the room)")
    c = o["chain"]
    print(f"   reached by: rays {c['rays']:.0%} -> wall masks {c['masks']:.0%} -> kept planes {c['planes']:.0%} -> a handed-off face {c['on_face']:.0%}")
    s = o["support"]
    print(f"-- our faces ({s['handed_off_m2']} m2, every {s['step_m'] * 100:.0f} cm): {s['supported']:.1%} have GT wall within {s['radius_m']} m; "
          "the rest is nearest to: " + ", ".join(f"{k} {v:.1%}" for k, v in list(s["unsupported_nearest_gt"].items())[:6]))
    print("-- per handed-off wall: GT wall near it | rays / masks / planes / on face | offset | GT top vs height | our face supported")
    for w in o["walls"]:
        if not w["gt_m2"]:
            print(f"   {w['wall']:6} no GT wall within {o['near_m']} m"); continue
        off = f"{w['offset_cm']:+5.1f} cm" if w["offset_cm"] is not None else "   -   "
        top = f"{w['gt_top_m']:.2f} vs {w['height_m']:.2f} ({w['height_minus_gt_cm']:+.0f} cm)" if w["gt_top_m"] else "-"
        print(f"   {w['wall']:6} {w['gt_m2']:5.2f} m2 | {w['chain_rays']:4.0%} / {w['chain_masks']:4.0%} / {w['chain_planes']:4.0%} / "
              f"{w['chain_on_face']:4.0%} | {off} | {top} | {w['supported']:.0%}")
    print(f"-- planes: mean |offset to GT wall points| {p['mean_abs_offset_cm']} cm over kept planes; kept planes with no GT wall: "
          f"{p['kept_without_gt'] or 'none'}")
    print("   " + ", ".join(f"{x['plane']} {x['offset_cm']:+.1f}" if x["offset_cm"] is not None else f"{x['plane']} -" for x in p["planes"]))
    print("-- room: " + "; ".join(f"{x['axis']} {x['ours_m']:.2f} vs GT {x['gt_m']:.2f} m ({x['diff_cm']:+.0f} cm)" for x in r["size"])
          + "  | sides inward: " + ", ".join(f"{x['side']} {x['inward_cm']:+.1f}" if x["inward_cm"] is not None else f"{x['side']} -" for x in r["sides"]))
    print(f"-- openings: GT {op['gt_found']}/{op['gt_total']} found, {op['false']} handed off with no GT")
    for x in op["handed_off"]:
        g = x["gt"]
        print(f"   {x['wall']} {x['width_m']:.2f} x {x['height_m']:.2f} m, z {x['z'][0]:.2f}-{x['z'][1]:.2f}"
              + (f" -> {g['name']}: centre {g['centre_cm']:+.0f}, width {g['width_cm']:+.0f}, bottom {g['bottom_cm']:+.0f}, top {g['top_cm']:+.0f} cm" if g else " -> no GT"))
    for g in op["gt"]:
        if not g["found"]:
            print(f"   missed: {g['name']} on {g['wall']}")
    print("-- thickness hints: " + "; ".join(f"{x['plane']} {x['kind']} {x['cm']} cm (GT {x['gt_cm']})" for x in t["hints"]))


def compare(cur: dict, base: dict) -> None:
    a, b = base["headline"], cur["headline"]
    print(f"== {cur['run']} (commit {cur['commit']}) vs baseline {base['run']} (commit {base['commit']})")
    moved = 0
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            moved += 1
            d = ""
            if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)):
                d = f"  ({b[k] - a[k]:+.4g})"
            print(f"   {k:40} {a.get(k)!s:>10} -> {b.get(k)!s:<10}{d}")
    print(f"   {moved} of {len(set(a) | set(b))} headline numbers moved")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="struct_v4_win5")
    ap.add_argument("--save-baseline", action="store_true", help=f"also write {BASELINE.name}")
    ap.add_argument("--compare", action="store_true", help="print the headline numbers that moved since the baseline")
    a = ap.parse_args()
    doc = evaluate(a.run)
    out = OUT_ROOT / a.run / "gt_eval.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float) + "\n")
    report(doc)
    print(f"wrote {out}")
    if a.save_baseline:
        BASELINE.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float) + "\n")
        print(f"wrote {BASELINE}")
    if a.compare:
        if not BASELINE.is_file():
            raise SystemExit(f"no baseline yet: run with --save-baseline first ({BASELINE})")
        compare(doc, json.loads(BASELINE.read_text()))


if __name__ == "__main__":
    main()
