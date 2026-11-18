"""Structure recognition (src.structure) against ScanNet GT: one fixed set of numbers per run.

    .venv/bin/python -m tools.structure.gt_eval --run struct_v4_win5                 # print, write gt_eval.json
    .venv/bin/python -m tools.structure.gt_eval --run struct_v4_win5 --save-baseline # also keep it as the baseline
    .venv/bin/python -m tools.structure.gt_eval --run struct_v4_win5 --compare       # this run vs the baseline

Evaluation only: nothing in src/ reads it. GT is the ScanNet annotation of scene0000_00 -
the 2D instance PNGs for the masks, and the wall faces, doors, windows and thicknesses that
pagegen.build_structure_gt measured on the annotated mesh (docs/assets/structure-gt/data.js),
moved into amb3r_floor through the oracle Sim(3) and axisAlignment.

Metrics, grouped by the step they judge:

  seg       2D segmentation: per label, the share of a mask's pixels on GT objects of its
            class (doorframe allowed in wall and door masks), mean and area-weighted; masks
            under 75%; frames whose GT shows a class that got no mask.
  coverage  per GT wall face, 10 cm cells (0.1-2.3 m high) with a ray endpoint, a wall-mask
            point, a point in a kept plane: where the evidence is lost.
  planes    per GT face, the kept plane on it: offset along the normal (+ = into the room),
            angle, length covered; GT faces with no plane; kept planes on no GT face.
  room      the closed rectangle per axis against the distance between the GT faces its
            opposite sides lie on; each side's offset.
  height    per kept plane, the highest ray surface on it against the GT face top; the
            storey height handed off against the GT tops.
  openings  each handed-off opening against the GT door / window / doorway on the same face
            (centre along the wall, width, bottom, top); GT openings found / missed.
  thickness the thickness hints against the GT measurements.

Results go to artifacts/structure/<run>/gt_eval.json; the baseline to
artifacts/structure/gt_eval_baseline.json. `--compare` prints every headline that moved.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from src.geometry.amb3r_mask_source import DEFAULT_SIM3
from src.geometry.floor_frame import Frame, scannet_scan_to_frame
from src.structure.frames import OUT_ROOT, RAYS, axis_alignment_for_eval, masks_dir

SCENE = config.SCANNET_ROOT / "scene0000_00"
GT_DATA = config.ROOT / "docs/assets/structure-gt/data.js"
HANDOFF = config.ARTIFACTS_ROOT / "handoff/scene0000_00/walls_harness"
BASELINE = OUT_ROOT / "gt_eval_baseline.json"
LABELS = ("wall", "floor", "door", "window")
RIGHT = {"wall": {"wall", "doorframe"}, "floor": {"floor"}, "door": {"door", "doorframe"}, "window": {"window"}}
PRESENT = {"wall": 0.01, "floor": 0.05, "door": 0.01, "window": 0.01}   # share of the frame for "the GT shows it"
LOW_PURITY = 0.75
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


# ---------------------------------------------------------------- 2D segmentation
def eval_segmentation(run: str) -> dict:
    seg = masks_dir(run)
    agg = json.loads((SCENE / "scene0000_00.aggregation.json").read_text())
    labels = {int(g["objectId"]): g["label"] for g in agg["segGroups"]}
    z = zipfile.ZipFile(SCENE / "scene0000_00_2d-instance.zip")
    rows, missing, frames_gt, frames_kept = [], [], Counter(), Counter()
    results = [json.loads(p.read_text()) for p in sorted(seg.glob("frame_*/result.json"))]
    for r in results:
        fid = r["frame_id"]
        g = np.asarray(Image.open(io.BytesIO(z.read(f"instance/{fid}.png")))).astype(int) - 1
        share = {c: float(np.isin(g, [i for i, x in labels.items() if x == c]).mean()) for c in LABELS}
        have = {k["label"] for k in r["kept"]}
        for c in LABELS:
            if share[c] >= PRESENT[c]:
                frames_gt[c] += 1
                if c in have:
                    frames_kept[c] += 1
                else:
                    missing.append({"frame": fid, "label": c, "gt_share": round(share[c], 3)})
        for k in r["kept"]:
            m = np.asarray(Image.open(seg / f"frame_{fid:06d}" / k["mask_file"]).convert("L")) > 0
            v = g[m]; v = v[v >= 0]
            by = Counter(labels.get(int(i), "?") for i in v)
            tot = max(1, sum(by.values()))
            purity = sum(n for c, n in by.items() if c in RIGHT[k["label"]]) / tot
            worst = max(((c, n / tot) for c, n in by.items() if c not in RIGHT[k["label"]]), key=lambda x: x[1], default=(None, 0.0))
            rows.append({"frame": fid, "label": k["label"], "plan_id": k["plan_id"], "where": k["where"],
                         "purity": round(purity, 4), "area": k["area_fraction"], "worst": [worst[0], round(worst[1], 3)]})
    per = {}
    for c in LABELS:
        rs = [x for x in rows if x["label"] == c]
        if rs:
            a = np.array([x["area"] for x in rs]); p = np.array([x["purity"] for x in rs])
            per[c] = {"masks": len(rs), "mean_purity": round(float(p.mean()), 4),
                      "area_weighted_purity": round(float((p * a).sum() / a.sum()), 4),
                      "frames_gt": frames_gt[c], "frames_with_mask": frames_kept[c],
                      "frame_recall": round(frames_kept[c] / frames_gt[c], 4) if frames_gt[c] else None}
    wins = [json.loads(p.read_text()) for p in sorted(seg.glob("window_*/window.json"))]
    return {"frames": len(results), "masks": len(rows), "per_label": per,
            "low_purity": sorted([x for x in rows if x["purity"] < LOW_PURITY], key=lambda x: x["purity"]),
            "missing": missing, "prompt_tokens": sum(w["usage"]["prompt_tokens"] for w in wins),
            "windows": len(wins)}


# ---------------------------------------------------------------- 3D
def eval_coverage(run: str, faces: list) -> list:
    d = OUT_ROOT / run
    g = np.load(d / "geom.npz")
    X = g["X"][g["wall_first"]].astype(np.float64); plane = g["wall_plane"]
    P = np.load(RAYS)["P"].astype(np.float64)
    nk = len(json.loads((d / "walls.json").read_text())["walls"])
    nz = int(round((BAND[1] - BAND[0]) / CELL))
    out = []
    for f in faces:
        c, n, u = (np.asarray(f[k]) for k in ("c", "n", "u"))
        L = f["a"][1] - f["a"][0]
        tot = max(1, int(round(L / CELL)) * nz)

        def cells(Q):
            dd = (Q[:, :2] - c) @ n; a = (Q[:, :2] - c) @ u
            s = (np.abs(dd) < 0.2) & (a > f["a"][0]) & (a < f["a"][1]) & (Q[:, 2] > BAND[0]) & (Q[:, 2] < BAND[1])
            return len(set(zip(np.floor(a[s] / CELL).astype(int), np.floor(Q[s, 2] / CELL).astype(int))))
        out.append({"face": f["key"], "name": f["name"], "length_m": round(L, 3),
                    "rays": round(min(1.0, cells(P) / tot), 4), "masks": round(min(1.0, cells(X) / tot), 4),
                    "kept": round(min(1.0, cells(X[(plane >= 0) & (plane < nk)]) / tot), 4)})
    return out


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


def eval_height(W: dict, C: dict, planes: dict, faces: list) -> dict:
    fmap = {f["key"]: f for f in faces}
    P = np.load(RAYS)["P"].astype(np.float64)
    rows = []
    for r in W["walls"]:
        c, n, u = (np.asarray(r[k]) for k in ("point", "normal", "along"))
        dd = (P[:, :2] - c) @ n; a = (P[:, :2] - c) @ u
        s = (np.abs(dd) < 0.12) & (a > r["a"][0] + 0.1) & (a < r["a"][1] - 0.1) & (P[:, 2] > 1.8)
        m = planes["matches"].get(r["id"])
        top = float(np.percentile(P[s, 2], 99)) if s.sum() > 20 else None
        gt = fmap[m["face"]]["z"][1] if m else None
        rows.append({"plane": r["id"], "face": m["face"] if m else None, "seen_top_m": round(top, 3) if top else None,
                     "gt_top_m": round(gt, 3) if gt else None,
                     "seen_minus_gt_cm": round((top - gt) * 100, 1) if top and gt else None})
    H = C["storey"]["height_m"]
    gts = [r["gt_top_m"] for r in rows if r["gt_top_m"]]
    return {"walls": rows, "storey_m": H, "storey_kind": C["storey"]["kind"],
            "storey_minus_gt_cm": [round((H - g) * 100, 1) for g in gts],
            "mean_abs_storey_error_cm": round(float(np.mean([abs(H - g) * 100 for g in gts])), 1) if gts else None}


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
           "seg": eval_segmentation(run), "coverage": eval_coverage(run, faces), "planes": planes,
           "room": eval_room(C, planes, faces), "height": eval_height(W, C, planes, faces),
           "openings": eval_openings(H, W, planes, faces), "thickness": eval_thickness(C, gaps, planes)}
    doc["headline"] = headline(doc)
    return doc


def headline(doc: dict) -> dict:
    """The few numbers to compare between runs; every one is in the report."""
    s, p, r, h, o = doc["seg"], doc["planes"], doc["room"], doc["height"], doc["openings"]
    cov = doc["coverage"]
    out = {f"seg.{c}.purity": v["mean_purity"] for c, v in s["per_label"].items()}
    out.update({f"seg.{c}.frame_recall": v["frame_recall"] for c, v in s["per_label"].items()})
    out.update({"seg.low_purity_masks": len(s["low_purity"]), "seg.masks": s["masks"], "seg.prompt_tokens": s["prompt_tokens"],
                "coverage.rays": round(float(np.mean([c["rays"] for c in cov])), 4),
                "coverage.masks": round(float(np.mean([c["masks"] for c in cov])), 4),
                "coverage.kept": round(float(np.mean([c["kept"] for c in cov])), 4),
                "planes.faces_with_plane": p["faces_with_plane"], "planes.faces_total": len(p["faces"]),
                "planes.mean_abs_offset_cm": p["mean_abs_offset_cm"], "planes.kept_on_no_face": len(p["kept_on_no_face"]),
                "room.area_m2": r["area_m2"], "height.mean_abs_storey_error_cm": h["mean_abs_storey_error_cm"],
                "openings.gt_found": o["gt_found"], "openings.gt_total": o["gt_total"], "openings.false": o["false"]})
    for x in r["size"]:
        out[f"room.size_{x['axis']}_diff_cm"] = x["diff_cm"]
    return out


# ---------------------------------------------------------------- report
def report(doc: dict) -> None:
    s, cov, p, r, h, o, t = (doc[k] for k in ("seg", "coverage", "planes", "room", "height", "openings", "thickness"))
    print(f"== {doc['run']}  (commit {doc['commit']}, scale {doc['frame']['source']} {doc['frame']['value']:.4f})")
    print(f"-- 2D segmentation: {s['masks']} masks in {s['frames']} frames, {s['prompt_tokens'] / 1e6:.2f} M prompt tokens")
    for c, v in s["per_label"].items():
        print(f"   {c:6} {v['masks']:3d} masks  purity {v['mean_purity']:.1%} (area-weighted {v['area_weighted_purity']:.1%})  "
              f"frames with mask {v['frames_with_mask']}/{v['frames_gt']}")
    print(f"   masks under {LOW_PURITY:.0%}: {len(s['low_purity'])}  "
          + "; ".join(f"{x['frame']} {x['label']} {x['purity']:.0%} ({x['worst'][0]} {x['worst'][1]:.0%})" for x in s["low_purity"][:6]))
    if s["missing"]:
        print("   GT shows it, no mask: " + ", ".join(f"{m['frame']} {m['label']} {m['gt_share']:.0%}" for m in s["missing"]))
    print("-- coverage of GT wall faces (10 cm cells, 0.1-2.3 m): rays / wall masks / kept planes")
    for c in cov:
        print(f"   {c['face']} {c['name']:<14} {c['length_m']:5.2f} m  {c['rays']:4.0%} / {c['masks']:4.0%} / {c['kept']:4.0%}")
    print(f"-- planes: {p['faces_with_plane']}/{len(p['faces'])} GT faces have a kept plane; mean |offset| {p['mean_abs_offset_cm']} cm; "
          f"kept planes on no GT face: {p['kept_on_no_face'] or 'none'}")
    for f in p["faces"]:
        if f["planes"]:
            print(f"   {f['face']} {f['name']:<14} {','.join(f['planes']):6} offset {f['offset_cm']:+6.1f} cm  angle {f['angle_deg']:4.1f}  "
                  f"covers {f['length_covered']:4.0%} of {f['gt_length_m']:.2f} m")
        else:
            print(f"   {f['face']} {f['name']:<14} -- no plane" + (f" (rejected: {','.join(f['rejected_on_it'])})" if f["rejected_on_it"] else ""))
    print("-- room: " + "; ".join(f"{x['axis']} {x['ours_m']:.2f} vs GT {x['gt_m']:.2f} m ({x['diff_cm']:+.0f} cm)" for x in r["size"])
          + "  | sides inward: " + ", ".join(f"{x['side']} {x['inward_cm']:+.1f}" if x["inward_cm"] is not None else f"{x['side']} -" for x in r["sides"]))
    print(f"-- height: storey {h['storey_m']:.2f} m ({h['storey_kind']}), mean |storey - GT top| {h['mean_abs_storey_error_cm']} cm")
    print("   " + ", ".join(f"{w['plane']}->{w['face']} seen {w['seen_top_m']} / GT {w['gt_top_m']}" for w in h["walls"]))
    print(f"-- openings: GT {o['gt_found']}/{o['gt_total']} found, {o['false']} handed off with no GT")
    for x in o["handed_off"]:
        g = x["gt"]
        print(f"   {x['wall']} {x['width_m']:.2f} x {x['height_m']:.2f} m, z {x['z'][0]:.2f}-{x['z'][1]:.2f}"
              + (f" -> {g['name']}: centre {g['centre_cm']:+.0f}, width {g['width_cm']:+.0f}, bottom {g['bottom_cm']:+.0f}, top {g['top_cm']:+.0f} cm" if g else " -> no GT"))
    for g in o["gt"]:
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
