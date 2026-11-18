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
MATCH_DEG, MATCH_OFF, MATCH_OVERLAP = 20.0, 0.40, 0.20
OPENING_LABELS = {"door": "door", "window": "window", "opening": "doorway"}


# ---------------------------------------------------------------- GT in amb3r_floor
def gt_to_frame() -> np.ndarray:
    """ScanNet axis-aligned frame (the GT page's) -> amb3r_floor."""
    S = scannet_scan_to_frame(Frame.load(), json.loads(DEFAULT_SIM3.read_text())["sim3_oracle_scale"])
    return S @ np.linalg.inv(axis_alignment_for_eval())


def gt_structure() -> dict:
    s = GT_DATA.read_text()
    return json.loads(s[s.index("{"):s.index(";\nwindow.STRUCT.pos")])


def gt_reference():
    """GT wall faces and thickness measurements in amb3r_floor: (faces, gaps)."""
    if not GT_DATA.is_file():
        return [], []
    gt = gt_structure()
    G = gt_to_frame()
    T = lambda p: G[:3, :3] @ np.asarray(p, dtype=np.float64) + G[:3, 3]
    R = G[:3, :3]
    faces = []
    for f in gt["faces"]:
        fc, fn, fu = T(f["c"]), R @ np.array(f["nh"]), R @ np.array(f["u"])
        faces.append({"key": f["key"], "name": f["name"], "c": fc[:2].tolist(),
                      "n": (fn[:2] / np.linalg.norm(fn[:2])).tolist(), "u": (fu[:2] / np.linalg.norm(fu[:2])).tolist(),
                      "a": f["a"], "corners": [T(q).tolist() for q in f["corners"]], "area_m2": f["area_m2"],
                      "z": [float(min(T(q)[2] for q in f["corners"])), float(max(T(q)[2] for q in f["corners"]))]})
    gaps = [{"name": g["name"], "a": g["a"], "b": g["b"], "cm": g["cm"], "spread_cm": g["spread_cm"],
             "p0": T(g["p0"]).tolist(), "p1": T(g["p1"]).tolist()} for g in gt.get("gaps", [])]
    return faces, gaps


def gt_openings(faces: list) -> list:
    """GT doors, windows and the inferred doorway, as spans along their host face (amb3r_floor)."""
    gt = gt_structure()
    G = gt_to_frame()
    fmap = {f["key"]: f for f in faces}
    out = []
    for it in gt["items"]:
        kind = OPENING_LABELS.get(it["label"])
        if kind is None or it["host"] not in fmap:
            continue
        f = fmap[it["host"]]
        C = np.array([G[:3, :3] @ np.asarray(q) + G[:3, 3] for q in it["corners"]])
        a = (C[:, :2] - f["c"]) @ np.asarray(f["u"])
        out.append({"key": it["key"], "name": it["name"], "kind": kind, "face": it["host"],
                    "a": [float(a.min()), float(a.max())], "z": [float(C[:, 2].min()), float(C[:, 2].max())],
                    "width_m": it["width_m"], "height_m": it["height_m"]})
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


def face_of_points(X: np.ndarray, faces: list) -> np.ndarray:
    """Each GT wall point's GT face (index into faces): the face plane it lies within 10 cm of, inside its extent."""
    out = np.full(len(X), -1); best = np.full(len(X), 0.10)
    for i, f in enumerate(faces):
        c, n, u = (np.asarray(f[k]) for k in ("c", "n", "u"))
        d = np.abs((X[:, :2] - c) @ n); a = (X[:, :2] - c) @ u
        ok = (d < best) & (a > f["a"][0] - 0.05) & (a < f["a"][1] + 0.05)
        out[ok], best[ok] = i, d[ok]
    return out


def match_gt(r: dict, faces: list) -> dict | None:
    """The GT face a plane lies on: facing within 20 deg, within 40 cm, overlapping 20 cm along it."""
    wc, wn, wu = np.array(r["point"]), np.array(r["normal"]), np.array(r["along"])
    best = None
    for f in faces:
        ang = float(np.degrees(np.arccos(np.clip(wn @ np.array(f["n"]), -1, 1))))
        if ang > MATCH_DEG:
            continue
        ends = [(wc + q * wu - f["c"]) @ np.array(f["u"]) for q in r["a"]]
        lo, hi = max(min(ends), f["a"][0]), min(max(ends), f["a"][1])
        off = float((wc + np.mean(r["a"]) * wu - f["c"]) @ np.array(f["n"]))
        if hi - lo < MATCH_OVERLAP or abs(off) > MATCH_OFF:
            continue
        if best is None or hi - lo > best["overlap_m"]:
            best = {"face": f["key"], "name": f["name"], "angle_deg": ang, "offset_cm": off * 100,
                    "overlap_m": float(hi - lo), "gt_length_m": float(f["a"][1] - f["a"][0]), "gt_area_m2": f["area_m2"]}
    return best


# ---------------------------------------------------------------- 3D
def eval_occupancy(run: str, H: dict, faces: list) -> dict:
    """GT wall surface against the handed-off solids, and the evidence chain that led to them."""
    gs = gt_surface()
    G = gs["points"][gs["wall"]]
    per_m2 = gs["per_m2"]
    solids = Solids(H)
    dist, which, in_hole = solids.distance(G)
    covered = (dist < TOL) & ~in_hole
    on_face = solids.face_distance(G) < TOL
    fidx = face_of_points(G, faces)
    # signed position of the GT surface relative to our room face (+ = behind it, our face sits into the room)
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
    chain = {"rays": reach(P), "masks": reach(Xw), "planes": reach(Xw[(plane >= 0) & (plane < nk)]), "handed_off": covered}
    rows = []
    for i, f in enumerate(faces + [{"key": "-", "name": "not on a GT face"}]):
        m = fidx == (i if f["key"] != "-" else -1)
        if not m.any():
            continue
        on = m & covered
        rows.append({"face": f["key"], "name": f["name"], "area_m2": round(m.sum() / per_m2, 2),
                     "completeness": round(float(on.sum() / m.sum()), 4),
                     "on_face": round(float((m & on_face).sum() / m.sum()), 4),
                     "offset_cm": round(float(np.median(q[on]) * 100), 1) if on.sum() > 50 else None,
                     "in_holes_m2": round(float((m & in_hole).sum() / per_m2), 3),
                     **{f"chain_{k}": round(float((v & m).sum() / m.sum()), 4) for k, v in chain.items()}})
    per_wall = []
    for k, w in enumerate(solids.walls):
        m = (which == k) & covered
        top = float(np.percentile(G[m, 2], 99)) if m.sum() > 50 else None
        per_wall.append({"wall": w["id"], "gt_m2": round(m.sum() / per_m2, 2), "height_m": w["h"],
                         "gt_top_m": round(top, 3) if top else None,
                         "height_minus_gt_cm": round((w["h"] - top) * 100, 1) if top else None,
                         "offset_cm": round(float(np.median(q[m]) * 100), 1) if m.sum() > 50 else None})
    # support: our faces, sampled, against the GT surface
    F, fk = solids.face_samples()
    dw, _ = cKDTree(G).query(F, distance_upper_bound=SUPPORT)
    supported = dw < SUPPORT
    da, ia = cKDTree(gs["points"]).query(F[~supported], distance_upper_bound=0.5)
    near = np.where(da < 0.5, gs["label"][np.minimum(ia, len(gs["label"]) - 1)], "nothing within 0.5 m")
    why = Counter(near.tolist())
    sup_wall = [{"wall": w["id"], "supported": round(float(supported[fk == k].mean()), 4) if (fk == k).any() else None}
                for k, w in enumerate(solids.walls)]
    return {"gt_wall_m2": round(len(G) / per_m2, 2), "tol_m": TOL,
            "completeness": round(float(covered.mean()), 4), "on_face": round(float(on_face.mean()), 4), "in_holes_m2": round(float(in_hole.sum() / per_m2), 3),
            "chain": {k: round(float(v.mean()), 4) for k, v in chain.items()},
            "median_offset_cm": round(float(np.nanmedian(q[covered]) * 100), 1),
            "faces": rows, "walls": per_wall,
            "support": {"step_m": FACE_STEP, "radius_m": SUPPORT, "samples": int(len(F)),
                        "supported": round(float(supported.mean()), 4), "handed_off_m2": round(len(F) * FACE_STEP ** 2, 2),
                        "unsupported_nearest_gt": {k: round(v / len(F), 4) for k, v in why.most_common()},
                        "walls": sup_wall}}


def eval_planes(W: dict, faces: list) -> dict:
    kept = {r["id"]: match_gt(r, faces) for r in W["walls"]}
    rej = {r["id"]: match_gt(r, faces) for r in W["rejected"]}
    by_face = {}
    for wid, m in kept.items():
        if m:
            by_face.setdefault(m["face"], []).append(wid)
    rows = []
    for f in faces:
        ids = by_face.get(f["key"], [])
        best = max(ids, key=lambda i: kept[i]["overlap_m"]) if ids else None
        m = kept[best] if best else None
        rows.append({"face": f["key"], "name": f["name"], "gt_length_m": round(f["a"][1] - f["a"][0], 3),
                     "planes": ids, "offset_cm": round(m["offset_cm"], 2) if m else None,
                     "angle_deg": round(m["angle_deg"], 2) if m else None,
                     "length_covered": round(sum(kept[i]["overlap_m"] for i in ids) / (f["a"][1] - f["a"][0]), 4) if ids else 0.0,
                     "rejected_on_it": [i for i, mm in rej.items() if mm and mm["face"] == f["key"]]})
    offs = [abs(r["offset_cm"]) for r in rows if r["offset_cm"] is not None]
    return {"faces": rows, "kept_on_no_face": [i for i, m in kept.items() if m is None],
            "faces_with_plane": sum(1 for r in rows if r["planes"]),
            "mean_abs_offset_cm": round(float(np.mean(offs)), 2) if offs else None,
            "matches": {k: v for k, v in kept.items()}}


def eval_room(C: dict, planes: dict, faces: list) -> dict:
    fmap = {f["key"]: f for f in faces}
    axes = [np.asarray(s["normal"]) for s in C["sides"]]
    sides = []
    for sd in C["sides"]:
        m = planes["matches"].get(sd["walls"][0]) if sd["walls"] else None
        if m:
            fc = np.asarray(fmap[m["face"]]["c"])
            sides.append({"side": sd["side"], "from": sd["from"], "face": m["face"], "name": m["name"],
                          "inward_cm": round((sd["offset"] - fc @ axes[sd["k"]]) * 100, 2)})
        else:
            sides.append({"side": sd["side"], "from": sd["from"], "face": None, "name": None, "inward_cm": None})
    by = {x["side"]: x for x in sides if x["face"]}
    size = []
    for k, (a, b, ours) in enumerate((("W", "E", C["rectangle"]["width_m"]), ("S", "N", C["rectangle"]["depth_m"]))):
        if a in by and b in by:
            ca, cb = (np.asarray(fmap[by[x]["face"]]["c"]) for x in (a, b))
            gt = abs((ca - cb) @ axes[k])
            size.append({"axis": f"{a}-{b}", "ours_m": round(ours, 4), "gt_m": round(gt, 4), "diff_cm": round((ours - gt) * 100, 2),
                         "faces": [by[a]["face"], by[b]["face"]]})
    return {"sides": sides, "size": size, "area_m2": round(C["rectangle"]["area_m2"], 3),
            "seen_outside_share": round(C["rectangle"]["seen_outside_share"], 4)}


def eval_openings(H: dict, W: dict, planes: dict, faces: list) -> dict:
    """Handed-off openings vs GT openings on the same face (matched by overlap along the wall)."""
    gto = gt_openings(faces)
    fmap = {f["key"]: f for f in faces}
    hw_face = {}
    for w in H["walls"]:
        ms = [planes["matches"].get(p) for p in w["evidence"]["planes"]]
        hw_face[w["id"]] = next((m["face"] for m in ms if m), None)
    found, rows = set(), []
    for w in H["walls"]:
        face = hw_face[w["id"]]
        for o in w["openings"]:
            row = {"wall": w["id"], "face": face, "from_m": o["from_m"], "to_m": o["to_m"], "z": o["z_floor"],
                   "width_m": o["width_m"], "height_m": o["height_m"], "gt": None}
            if face:
                f = fmap[face]
                q = np.array([o["corners_floor"][0][:2], o["corners_floor"][1][:2]])
                a = sorted(((q - f["c"]) @ np.asarray(f["u"])).tolist())
                best = None
                for g in gto:
                    if g["face"] != face:
                        continue
                    ov = min(a[1], g["a"][1]) - max(a[0], g["a"][0])
                    if ov > 0 and (best is None or ov > best[0]):
                        best = (ov, g)
                if best:
                    g = best[1]; found.add(g["key"])
                    row["gt"] = {"key": g["key"], "name": g["name"], "kind": g["kind"],
                                 "centre_cm": round(((a[0] + a[1]) - (g["a"][0] + g["a"][1])) / 2 * 100, 1),
                                 "width_cm": round((a[1] - a[0] - (g["a"][1] - g["a"][0])) * 100, 1),
                                 "bottom_cm": round((o["z_floor"][0] - g["z"][0]) * 100, 1),
                                 "top_cm": round((o["z_floor"][1] - g["z"][1]) * 100, 1)}
            rows.append(row)
    # a door leaf and its frame are one opening: count GT openings by face and overlap
    groups = []
    for g in sorted(gto, key=lambda g: (g["face"], g["a"][0])):
        if groups and groups[-1]["face"] == g["face"] and g["a"][0] < groups[-1]["a"][1]:
            groups[-1]["keys"].append(g["key"]); groups[-1]["a"][1] = max(groups[-1]["a"][1], g["a"][1])
        else:
            groups.append({"face": g["face"], "a": list(g["a"]), "keys": [g["key"]], "name": g["name"], "kind": g["kind"]})
    for gr in groups:
        gr["found"] = any(k in found for k in gr["keys"])
    return {"handed_off": rows, "gt": groups, "gt_found": sum(g["found"] for g in groups), "gt_total": len(groups),
            "false": sum(1 for r in rows if r["gt"] is None)}


def eval_thickness(C: dict, gaps: list, planes: dict) -> dict:
    hints = []
    for r in C["walls"]:
        m = planes["matches"].get(r["id"])
        for h in r.get("thickness_hint", []):
            v = h.get("gap_cm", h.get("median_cm"))
            gt = None
            if m:
                for g in gaps:
                    if g["a"] == m["face"] and ((h["kind"] == "door_recess") == g["b"].startswith("door#")):
                        gt = g["cm"]
            hints.append({"plane": r["id"], "kind": h["kind"], "cm": round(v, 1), "gt_cm": gt,
                          "diff_cm": round(v - gt, 1) if gt is not None else None})
    return {"hints": hints, "assumed_m": C["thickness"], "gt_measured_cm": [g["cm"] for g in gaps]}


# ---------------------------------------------------------------- run
def evaluate(run: str) -> dict:
    d = OUT_ROOT / run
    W = json.loads((d / "walls.json").read_text())
    C = json.loads((d / "closed.json").read_text())
    H = json.loads((HANDOFF / "walls.json").read_text())
    faces, gaps = gt_reference()
    planes = eval_planes(W, faces)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    except OSError:
        commit = None
    doc = {"run": run, "commit": commit, "frame": Frame.load().doc["scale"],
           "occupancy": eval_occupancy(run, H, faces), "planes": planes, "room": eval_room(C, planes, faces),
           "openings": eval_openings(H, W, planes, faces), "thickness": eval_thickness(C, gaps, planes)}
    doc["headline"] = headline(doc)
    return doc


def headline(doc: dict) -> dict:
    """The few numbers to compare between runs; every one is in the report."""
    o, r, op = doc["occupancy"], doc["room"], doc["openings"]
    hs = [abs(w["height_minus_gt_cm"]) for w in o["walls"] if w["height_minus_gt_cm"] is not None]
    out = {"wall.completeness": o["completeness"], "wall.on_face": o["on_face"], "wall.support": o["support"]["supported"],
           "wall.median_offset_cm": o["median_offset_cm"], "wall.gt_in_holes_m2": o["in_holes_m2"],
           "wall.mean_abs_height_error_cm": round(float(np.mean(hs)), 1) if hs else None,
           **{f"chain.{k}": v for k, v in o["chain"].items()},
           **{f"face.{f['face']}.on_face": f["on_face"] for f in o["faces"] if f["face"] != "-"},
           "openings.gt_found": op["gt_found"], "openings.gt_total": op["gt_total"], "openings.false": op["false"],
           "room.area_m2": r["area_m2"]}
    for x in r["size"]:
        out[f"room.size_{x['axis']}_diff_cm"] = x["diff_cm"]
    return out


# ---------------------------------------------------------------- report
def report(doc: dict) -> None:
    o, p, r, op, t = (doc[k] for k in ("occupancy", "planes", "room", "openings", "thickness"))
    print(f"== {doc['run']}  (commit {doc['commit']}, scale {doc['frame']['source']} {doc['frame']['value']:.4f})")
    print(f"-- GT wall surface {o['gt_wall_m2']} m2: {o['completeness']:.1%} inside a handed-off wall solid, "
          f"{o['on_face']:.1%} within {o['tol_m']} m of a room-side face; "
          f"{o['in_holes_m2']} m2 inside a handed-off opening; median offset {o['median_offset_cm']:+.1f} cm (+ = our face into the room)")
    c = o["chain"]
    print(f"   reached by: rays {c['rays']:.0%} -> wall masks {c['masks']:.0%} -> kept planes {c['planes']:.0%} -> handed-off walls {c['handed_off']:.0%}")
    print("   per GT face (area | rays / masks / planes / in a solid | on our face | offset):")
    for f in o["faces"]:
        off = f"{f['offset_cm']:+.1f} cm" if f["offset_cm"] is not None else "-"
        print(f"   {f['face']} {f['name']:<14} {f['area_m2']:5.2f} m2 | {f['chain_rays']:4.0%} / {f['chain_masks']:4.0%} / "
              f"{f['chain_planes']:4.0%} / {f['chain_handed_off']:4.0%} | {f['on_face']:4.0%} | {off}")
    s = o["support"]
    print(f"-- our faces ({s['handed_off_m2']} m2, every {s['step_m'] * 100:.0f} cm): {s['supported']:.1%} have GT wall within {s['radius_m']} m; "
          "the rest is nearest to: " + ", ".join(f"{k} {v:.1%}" for k, v in list(s["unsupported_nearest_gt"].items())[:6]))
    print("   per wall: " + ", ".join(f"{w['wall']} {w['supported']:.0%}" for w in s["walls"]))
    print("-- per handed-off wall: GT surface on it, GT top vs height_m, offset")
    for w in o["walls"]:
        print(f"   {w['wall']:6} {w['gt_m2']:5.2f} m2  GT top {w['gt_top_m']} vs {w['height_m']:.2f} m "
              f"({w['height_minus_gt_cm']:+.0f} cm)  offset {w['offset_cm']:+.1f} cm" if w["gt_top_m"] else f"   {w['wall']:6} no GT wall on it")
    print("-- room: " + "; ".join(f"{x['axis']} {x['ours_m']:.2f} vs GT {x['gt_m']:.2f} m ({x['diff_cm']:+.0f} cm)" for x in r["size"]))
    print(f"-- openings: GT {op['gt_found']}/{op['gt_total']} found, {op['false']} handed off with no GT")
    for x in op["handed_off"]:
        g = x["gt"]
        print(f"   {x['wall']} {x['width_m']:.2f} x {x['height_m']:.2f} m, z {x['z'][0]:.2f}-{x['z'][1]:.2f}"
              + (f" -> {g['name']}: centre {g['centre_cm']:+.0f}, width {g['width_cm']:+.0f}, bottom {g['bottom_cm']:+.0f}, top {g['top_cm']:+.0f} cm" if g else " -> no GT"))
    for g in op["gt"]:
        if not g["found"]:
            print(f"   missed: {g['name']} on {g['face']}")
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
