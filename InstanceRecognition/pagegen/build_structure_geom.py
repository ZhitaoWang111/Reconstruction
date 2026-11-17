"""Pack the wall-geometry outputs for docs/structure-geom-3d.html (window.GEOM in its data.js).

    python -m pagegen.build_structure_geom --run struct_v4_win5

Reads only what `src.structure` and the wall handoff wrote:

* artifacts/structure/<run>/points.npz, masks.json   (src.structure.lift)
* artifacts/structure/<run>/geom.npz, walls.json     (src.structure.walls)
* artifacts/structure/<run>/closed.json              (src.structure.close)
* artifacts/structure/scene_rays.npz                 (src.structure.rays; the page's background)
* artifacts/handoff/scene0000_00/walls_harness/walls.json (the handoff: footprints and wall ids,
  so the thickness drawn on the page is the one handed off, corners included)

The GT comparison is display only: the ScanNet wall faces and measured thicknesses are read
from docs/assets/structure-gt/data.js (built from the annotation by the GT page) and moved
into the floor frame. Nothing here feeds back into the walls.

Points are thinned on a voxel grid per label and quantised to uint16 over the scene bounds.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from src import config
from src.structure.frames import OUT_ROOT, RAYS, FloorFrame, rotation_to_z

DOCS = config.ROOT / "docs"
DEFAULT_OUT = DOCS / "assets/structure-geom/data.js"
GT_DATA = DOCS / "assets/structure-gt/data.js"
HANDOFF = config.ARTIFACTS_ROOT / "handoff/scene0000_00/walls_harness/walls.json"
VOXEL = {0: 0.02, 1: 0.03, 2: 0.015, 3: 0.01}        # wall floor door window
BG_VOXEL = 0.04                                      # the whole scene, every pixel of every frame
SIDE_OF = {0: "W", 1: "S", 2: "E", 3: "N"}
WHY_ZH = {"support area below the floor": "支持面积不足", "seen in one frame only": "只有一帧支持"}   # the page is in Chinese


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def thin(xyz, *cols, v):
    key = np.floor(xyz / v).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return (xyz[first], *[c[first] for c in cols])


def gt_reference(floor: dict):
    """ScanNet wall faces and thickness measurements, moved into the floor frame. Display only."""
    if not GT_DATA.is_file():
        return [], []
    s = GT_DATA.read_text()
    gt = json.loads(s[s.index("{"):s.index(";\nwindow.STRUCT.pos")])
    T = FloorFrame(floor)
    R = rotation_to_z(np.asarray(floor["up_in_lift_frame"], dtype=np.float64))
    faces = []
    for f in gt["faces"]:
        fc, fn, fu = T(f["c"]), R @ np.array(f["nh"]), R @ np.array(f["u"])
        faces.append({"key": f["key"], "name": f["name"], "c": fc[:2].tolist(),
                      "n": (fn[:2] / np.linalg.norm(fn[:2])).tolist(), "u": (fu[:2] / np.linalg.norm(fu[:2])).tolist(),
                      "a": f["a"], "corners": [T(q).tolist() for q in f["corners"]], "area_m2": f["area_m2"]})
    gaps = [{"name": g["name"], "a": g["a"], "b": g["b"], "cm": g["cm"], "spread_cm": g["spread_cm"],
             "p0": T(g["p0"]).tolist(), "p1": T(g["p1"]).tolist()} for g in gt.get("gaps", [])]
    return faces, gaps


def match_gt(r: dict, faces: list) -> dict | None:
    """The GT face a plane lies on: facing within 20 deg, within 40 cm, overlapping 20 cm along it."""
    wc, wn, wu = np.array(r["point"]), np.array(r["normal"]), np.array(r["along"])
    best = None
    for f in faces:
        ang = float(np.degrees(np.arccos(np.clip(wn @ np.array(f["n"]), -1, 1))))
        if ang > 20:
            continue
        ends = [(wc + q * wu - f["c"]) @ np.array(f["u"]) for q in r["a"]]
        lo, hi = max(min(ends), f["a"][0]), min(max(ends), f["a"][1])
        off = float((wc + np.mean(r["a"]) * wu - f["c"]) @ np.array(f["n"]))
        if hi - lo < 0.2 or abs(off) > 0.4:
            continue
        if best is None or hi - lo > best["overlap_m"]:
            best = {"face": f["key"], "name": f["name"], "angle_deg": ang, "offset_cm": off * 100,
                    "overlap_m": float(hi - lo), "gt_length_m": float(f["a"][1] - f["a"][0]), "gt_area_m2": f["area_m2"]}
    return best


def closed_with_handoff(closed: dict, handoff_path: Path) -> dict:
    """The closure as the page draws it: each final wall gets the handed-off id and footprint.

    close.json describes walls by their pieces; the handoff adds the thickness footprint with
    corners joined (mitred exterior corners, interior walls butted onto the exterior face).
    Matched by side for exterior walls and by source plane for interior ones.
    """
    if not handoff_path.is_file():
        return closed
    handed = json.loads(handoff_path.read_text())["walls"]
    by_side = {w["id"]: w for w in handed if w["role"] == "exterior"}
    by_plane = {w["evidence"]["planes"][0]: w for w in handed if w["role"] == "interior"}
    for f in closed["final"]:
        h = by_side.get(f"ext_{SIDE_OF[f['side']]}") if f["role"] == "exterior" else by_plane.get(f["walls"][0])
        if h is None:
            continue
        f["id"] = h["id"]
        f["footprint"] = h["footprint_floor"]
        f["thickness_m"] = h["thickness_m"]
    closed["handoff"] = str(handoff_path)
    return closed


def build(run: str, out: Path = DEFAULT_OUT, handoff: Path = HANDOFF) -> dict:
    d = OUT_ROOT / run
    p = np.load(d / "points.npz")
    g = np.load(d / "geom.npz")
    W = json.loads((d / "walls.json").read_text())
    masks = json.loads((d / "masks.json").read_text())
    closed = json.loads((d / "closed.json").read_text()) if (d / "closed.json").is_file() else None
    X = g["X"].astype(np.float64)                        # floor frame: z up, floor at z = 0
    N = g["N"].astype(np.float64)
    lab, frame, mask, rgb = p["label"], p["frame"], p["mask"], p["rgb"]
    planes = W["walls"] + W["rejected"]

    # ---- points: thinned per label; a wall point goes to the nearest plane within 7.5 cm and its extent
    pts, cats, plane_of, fr, mk, rgs = [], [], [], [], [], []
    for L in (0, 1, 2, 3):
        sel = lab == L
        x, f, m, c = thin(X[sel], frame[sel], mask[sel], rgb[sel], v=VOXEL[L])
        pl = np.zeros(len(x), np.int16)
        if L == 0:
            best = np.full(len(x), 1e9); pl = np.full(len(x), -1, np.int16)
            for i, r in enumerate(planes):
                u, nn = np.array(r["along"]), np.array(r["normal"])
                dd = (x[:, :2] - np.array(r["point"])) @ nn
                t = (x[:, :2] - np.array(r["point"])) @ u
                ok = ((np.abs(dd) < 0.075) & (t > r["a"][0] - 0.05) & (t < r["a"][1] + 0.05)
                      & (x[:, 2] > r["z"][0] - 0.1) & (x[:, 2] < r["z"][1] + 0.1))
                hit = ok & (np.abs(dd) < best)
                best[hit] = np.abs(dd[hit]); pl[hit] = i
        pts.append(x); cats.append(np.full(len(x), L, np.uint16)); plane_of.append(pl)
        fr.append(f); mk.append(m); rgs.append(c)
    X0 = np.concatenate(pts).astype(np.float32); CAT = np.concatenate(cats); PL = np.concatenate(plane_of)
    FR = np.concatenate(fr).astype(np.uint16); MK = np.concatenate(mk).astype(np.uint16); RGB = np.concatenate(rgs)

    # ---- planes, each with its GT face for the comparison columns
    faces, gaps = gt_reference(W["floor"])
    walls = []
    for r in planes:
        wc, wu = np.array(r["point"]), np.array(r["along"])
        rec = {k: r[k] for k in ("id", "point", "normal", "along", "a", "z", "area_m2", "frames", "masks", "rms_cm",
                                 "heading_deg", "length_m")}
        for k in ("openings", "seen_share", "solid_share", "line"):
            if k in r:
                rec[k] = r[k]
        rec["kind"] = "kept" if r["id"].startswith("W") else "rejected"
        rec["why"] = WHY_ZH.get(r.get("why", ""), r.get("why", ""))
        rec["corners"] = [[*(wc + r["a"][0] * wu), r["z"][0]], [*(wc + r["a"][1] * wu), r["z"][0]],
                          [*(wc + r["a"][1] * wu), r["z"][1]], [*(wc + r["a"][0] * wu), r["z"][1]]]
        rec["gt"] = match_gt(r, faces)
        walls.append(rec)

    # ---- background: the scene rays, in the same floor frame
    z = np.load(RAYS)
    B, Brgb = thin(FloorFrame(W["floor"])(z["P"].astype(np.float64)), z["rgb"], v=BG_VOXEL)
    blo, bhi = B.min(0), B.max(0)
    bq = np.round((B - blo) / np.maximum(bhi - blo, 1e-6) * 65535).astype(np.uint16)

    lo, hi = X0.min(0), X0.max(0)
    q = np.round((X0 - lo) / np.maximum(hi - lo, 1e-6) * 65535).astype(np.uint16)
    doc = {"run": run, "sources": {"structure": str(d), "rays": str(RAYS), "handoff": str(handoff)},
           "unit": "m", "axis": "floor frame: z up, z=0 at the floor", "n": int(len(X0)),
           "lo": lo.tolist(), "hi": hi.tolist(), "pos": b64(q), "rgb": b64(RGB.astype(np.uint8)),
           "cat": b64(CAT), "plane": b64(PL), "frame": b64(FR), "mask": b64(MK),
           "masks": masks, "cams": [{"frame": int(f), "p": np.array(c).tolist()} for f, c in zip(p["cam_frames"], g["cams"])],
           "floor": W["floor"], "params": W["params"], "walls": walls, "cuts": W.get("cuts", []),
           "scene_rays": W.get("scene_rays"), "scene_frames": W.get("scene_frames"), "lines": W.get("lines"),
           "local_planes": W.get("local_planes"), "unassigned": W.get("wall_points_unassigned"),
           "horizontal": W.get("wall_points_horizontal"), "gt_faces": faces, "gt_gaps": gaps,
           "counts": {str(L): int((CAT == L).sum()) for L in range(4)},
           "closed": closed_with_handoff(closed, handoff) if closed else None,
           "bg": {"n": int(len(B)), "lo": blo.tolist(), "hi": bhi.tolist(), "pos": b64(bq),
                  "rgb": b64(Brgb.astype(np.uint8)), "voxel_m": BG_VOXEL, "rays": int(len(z["P"])),
                  "frames": int(len(z["cams"]))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.GEOM=" + json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"{len(X0)} points, {len(B)} background points, {len(walls)} planes -> {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print("planes -> GT:", [(r["id"], r["gt"]["face"] if r["gt"] else None) for r in walls])
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="struct_v4_win5")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--handoff", type=Path, default=HANDOFF)
    a = ap.parse_args()
    build(a.run, a.out, a.handoff)


if __name__ == "__main__":
    main()
