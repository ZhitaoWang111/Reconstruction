"""Steps 2-3: check the floor, then the wall planes.

    python -m src.structure.walls --run struct_v4_win5

Step 2  The points arrive in the amb3r_floor frame (src.geometry.floor_frame), which was
        fitted to the floor masks: z = 0 is the floor and +z is up. Here the floor points
        are only measured against z = 0 (residual, tilt, area) so a frame fitted to another
        run's masks shows up.
Step 3  Wall planes, assuming walls are flat and vertical:

  3a  Local planes, one mask at a time. A mask's wall points are split by their own facing
      (a mask running round a corner gives one set per face), then lines are pulled out of
      the top view one at a time and each is refitted with outliers trimmed.
  3b  Wall lines: local planes facing the same way (within FACING_DEG) whose offsets chain
      within DRIFT are one line, however far apart along the wall their masks are.
  3c  The ray test cuts each line. Every pixel of every frame is a ray from a camera to a
      surface (rays.py); along a wall line, rays that end on it mean something is there,
      rays that pass through it mean nothing is. A column passed through from floor to top
      with nothing solid in it cuts the line; see-through cells under a solid top are an
      opening in the same wall.
  3d  Each piece is a wall plane; it is kept if MIN_AREA of surface in MIN_FRAMES frames
      supports it.

The masks say where wall surface was seen; the rays say what lies between two pieces of
it, which is what replaced the gap and merge lengths the first two versions needed.
Parameters left: DRIFT (one surface's depth can disagree by up to ~24 cm between frames
here; results do not change between 15 and 70 cm), FACING_DEG, and the support floor.

Writes `walls.json` (floor check, kept and rejected planes with their openings, ray cuts)
and `geom.npz` (points in the amb3r_floor frame, and the plane each thinned wall point went to).
No annotation is read.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import ndimage

from src.geometry.floor_frame import Frame

from .frames import RAYS, Rays, run_dir

rng = np.random.default_rng(0)

VOXEL = 0.03          # m; thinning grid, and the area one thinned point stands for
TOL = 0.05            # m; RANSAC inlier band within one mask
SPLIT_DEG = 30.0      # a mask whose normals fall into two directions further apart than this is split
DRIFT = 0.30          # m; see the docstring
FACING_DEG = 12.0
CELL = 0.10           # m; grid on the wall plane for the ray test
MIN_RAYS = 5          # rays in a cell before it counts
MIN_LOCAL = 0.08      # m2; a local plane worth keeping
MIN_AREA = 0.40       # m2 of supporting surface for a wall
MIN_FRAMES = 2
MIN_OPENING = 0.25    # m2 of see-through cells inside a wall to report an opening
HORIZ_DEG = 25.0      # a wall point's own normal must be within this of horizontal
Z_LO, Z_HI = 0.10, 2.70


def fit_plane(p):
    c = p.mean(0)
    return c, np.linalg.eigh(np.cov((p - c).T))[1][:, 0]


def ransac_plane(p, tol, iters=400):
    best = None
    for _ in range(iters):
        s = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); length = np.linalg.norm(n)
        if length < 1e-9:
            continue
        n /= length
        inl = np.abs((p - s[0]) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    return best


def room_axis(normals) -> float:
    """The room's main horizontal direction (radians, mod 90 deg) from wall normals: each points
    at its angle times four, so walls at right angles to each other agree."""
    n = np.asarray(normals, dtype=np.float64)[:, :2]
    n = n[np.linalg.norm(n, axis=1) > 0.5]
    return float(np.angle(np.exp(4j * np.arctan2(n[:, 1], n[:, 0])).sum()) / 4)


def voxel_thin(xyz, *cols, axis: float = 0.0):
    """One point per VOXEL cube, the cubes turned to the room's axis.

    A thinned point stands for VOXEL**2 of surface, which holds only for a wall parallel to the
    cubes' sides: a wall at 35 deg to them crosses more cubes and comes out up to 13% larger here.
    Turning the cubes to the room's own axis keeps that from depending on where +x happens to point.
    """
    c, s = np.cos(axis), np.sin(axis)
    q = np.c_[xyz[:, 0] * c + xyz[:, 1] * s, -xyz[:, 0] * s + xyz[:, 1] * c, xyz[:, 2]]
    key = np.floor(q / VOXEL).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return (xyz[first], *[c_[first] for c_ in cols], first)


def line_ransac(xy, tol, iters=600):
    """The line in the top view with the most points within tol: the best vertical plane."""
    best = None
    for _ in range(iters):
        a, b = xy[rng.choice(len(xy), 2, replace=False)]
        d = b - a; length = np.linalg.norm(d)
        if length < 0.2:
            continue
        n = np.array([-d[1], d[0]]) / length
        inl = np.abs((xy - a) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    return best


def fit_vertical(xy):
    c = xy.mean(0)
    _, v = np.linalg.eigh(np.cov((xy - c).T))
    return c, v[:, 0], v[:, 1]


def robust_vertical(P, N, idx, rounds=3):
    """Vertical plane through P[idx], trimming points off it (max(3 cm, 2 sigma)). Returns (c, n, kept idx)."""
    keep = idx.copy()
    c, n, _ = fit_vertical(P[keep, :2])
    if (N[keep, :2] @ n).mean() < 0:
        n = -n
    for _ in range(rounds):
        d = (P[keep, :2] - c) @ n
        sig = 1.4826 * np.median(np.abs(d - np.median(d)))
        k = np.abs(d) < max(0.03, 2 * sig)
        if k.all() or k.sum() < 30:          # < 30: two surfaces with nothing between them; leave the set
            break
        keep = keep[k]
        c, n, _ = fit_vertical(P[keep, :2])
        if (N[keep, :2] @ n).mean() < 0:
            n = -n
    return c, n, keep


def opening_bounds(solid, free, i0, i1, j0, j1):
    """What closes a see-through block of cells [i0, i1) x [j0, j1) on each side.

    left / right / top: "solid" when most cells just outside that side are wall the rays stop on,
    "extent_end" when that side is the end of the observed wall (the grid stops there: the wall
    may simply end), "unseen" otherwise.
    bottom: "solid" when wall closes it from below (a window sill), "floor" when nothing solid lies
    between the block and the floor - rays aimed that low hit the floor in front of the wall first,
    so the cells under a doorway are unseen rather than open; the doorway reaches the floor.
    """
    hard = solid >= MIN_RAYS
    seen = hard | free

    def side(cells_hard, cells_seen, at_edge):
        if at_edge:
            return "extent_end"
        n = cells_hard.size
        return "solid" if n and cells_hard.sum() >= 0.5 * n else "unseen"

    na = solid.shape[0]
    left = side(hard[i0 - 1, j0:j1], seen[i0 - 1, j0:j1], i0 == 0) if i0 > 0 else "extent_end"
    right = side(hard[i1, j0:j1], seen[i1, j0:j1], False) if i1 < na else "extent_end"
    top = side(hard[i0:i1, j1:].any(1), None, j1 >= solid.shape[1]) if j1 < solid.shape[1] else "extent_end"
    under = hard[i0:i1, :j0]
    bottom = "floor" if (j0 == 0 or not under.any(1).mean() >= 0.5) else "solid"
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def ray_grid(rays, c, n, lo, hi):
    return rays.grid(c, n, lo, hi, CELL, DRIFT, Z_LO, Z_HI)


def floor_check(xyz, frame, mask, lab, frame_doc: dict) -> dict:
    """The floor points measured against z = 0 of the frame they arrive in."""
    f = xyz[lab == 1]
    c, n = fit_plane(f[rng.choice(len(f), min(len(f), 80000), replace=False)])
    res = np.abs(f[:, 2])
    return {"frame": frame_doc["file"], "frame_floor_fit": frame_doc["floor_fit"],
            "tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(n[2]))))),
            "plane_z_m": float(c[2]),
            "residual_median_cm": float(np.median(res) * 100), "residual_p90_cm": float(np.percentile(res, 90) * 100),
            "within_3cm": float(np.mean(res < 0.03)), "points": int(len(f)),
            "masks": int(len(np.unique(mask[lab == 1]))), "frames": int(len(np.unique(frame[lab == 1])))}


def walls(run: str, dump: bool = False, trace: bool = False) -> dict:
    d = run_dir(run)
    pts = np.load(d / "points.npz")
    xyz, nrm, lab = pts["xyz"].astype(np.float64), pts["normal"].astype(np.float64), pts["label"]
    frame, mask = pts["frame"], pts["mask"]

    # ---- step 2: the floor, as the frame left it
    F = Frame.load()
    X, N, C = xyz, nrm, pts["cam_xyz"].astype(np.float64)
    floor = floor_check(X, frame, mask, lab, F.doc)
    w = lab == 0
    axis = room_axis(N[w][np.abs(N[w][:, 2]) < np.sin(np.radians(HORIZ_DEG))])
    fz = X[lab == 1]
    floor["area_m2"] = float(len(voxel_thin(fz, axis=axis)[0]) * VOXEL ** 2)
    print(f"floor: {floor['tilt_deg']:.2f} deg off +z, at z = {floor['plane_z_m'] * 100:+.1f} cm, residual median "
          f"{floor['residual_median_cm']:.1f} cm, about {floor['area_m2']:.1f} m2")

    # ---- step 3: wall points, thinned on cubes turned to the room's axis
    P, Nw, Fw, Mw, first = voxel_thin(X[w], N[w], frame[w], mask[w], axis=axis)
    print(f"thinning grid turned to the room axis, {np.degrees(axis):+.2f} deg")
    horiz = np.abs(Nw[:, 2]) < np.sin(np.radians(HORIZ_DEG))
    rays = Rays(RAYS, F)
    print(f"rays: {len(rays.P)} from {rays.frames} frames")

    # ---- 3a: local planes, one mask at a time, then one sweep over what the masks left
    def extract(idx, out):
        used = np.zeros(len(idx), bool)
        for _ in range(40):
            pos = np.flatnonzero(~used); cur = idx[pos]
            if len(cur) * VOXEL ** 2 < MIN_LOCAL:
                break
            inl = line_ransac(P[cur, :2], TOL)
            if inl is None or inl.sum() * VOXEL ** 2 < MIN_LOCAL:
                break
            c2, n2, keep = robust_vertical(P, Nw, cur[inl])
            out.append({"c": c2, "n": n2, "idx": keep})
            used[pos[inl]] = True

    local = []
    for m in np.unique(Mw[horiz]):
        mi = np.flatnonzero(horiz & (Mw == m))
        if len(mi) * VOXEL ** 2 < MIN_LOCAL:
            continue
        n2 = Nw[mi, :2] / np.maximum(np.linalg.norm(Nw[mi, :2], axis=1, keepdims=True), 1e-9)
        z = np.exp(2j * np.arctan2(n2[:, 1], n2[:, 0]))
        near = np.abs(np.angle(z / z.mean())) < np.radians(SPLIT_DEG)
        both = near.sum() * VOXEL ** 2 >= MIN_LOCAL and (~near).sum() * VOXEL ** 2 >= MIN_LOCAL
        for part in ([mi[near], mi[~near]] if both else [mi]):
            extract(part, local)
    n_mask = len(local)
    covered = np.zeros(len(P), bool)
    for r in local:
        covered[r["idx"]] = True
    extract(np.flatnonzero(horiz & ~covered), local)
    print(f"local planes: {n_mask} from {len(np.unique(Mw[horiz]))} masks, {len(local) - n_mask} from the sweep")
    for k, r in enumerate(local):
        r["k"], r["from_sweep"] = k, k >= n_mask

    # ---- 3b: wall lines
    reps = []
    for r in local:
        ang = np.arctan2(r["n"][1], r["n"][0])
        hit = next((j for j, a0 in enumerate(reps) if abs(np.angle(np.exp(1j * (ang - a0)))) <= np.radians(FACING_DEG)), None)
        if hit is None:
            reps.append(ang); hit = len(reps) - 1
        r["rep"] = hit
    lines = []
    for rep in range(len(reps)):
        g = [r for r in local if r["rep"] == rep]
        n0 = np.array([np.cos(reps[rep]), np.sin(reps[rep])])
        g.sort(key=lambda r: float(r["c"] @ n0))
        cur = [g[0]]
        for r0, r1 in zip(g, g[1:]):
            if float(r1["c"] @ n0) - float(r0["c"] @ n0) <= DRIFT:
                cur.append(r1)
            else:
                lines.append(cur); cur = [r1]
        lines.append(cur)
    print(f"wall lines: {len(lines)}")

    # ---- 3c: cut each line where the rays see through it
    pieces, cuts, line_trace = [], [], []
    for li, g in enumerate(lines):
        idx = np.unique(np.concatenate([r["idx"] for r in g]))
        if len(idx) * VOXEL ** 2 < MIN_LOCAL:
            continue
        c, n, idx = robust_vertical(P, Nw, idx)
        u = np.array([-n[1], n[0]])
        al = (P[idx, :2] - c) @ u
        lo, hi = al.min() - CELL, al.max() + CELL
        solid, through = ray_grid(rays, c, n, lo, hi)
        free = (through >= MIN_RAYS) & (through > 3 * solid)
        hard = solid >= MIN_RAYS
        sup = np.zeros(solid.shape[0], bool)
        sup[np.clip(np.floor((al - lo) / CELL).astype(int), 0, len(sup) - 1)] = True
        open_col = (free.sum(1) >= 3) & (hard.sum(1) == 0) & ~sup
        runs, _ = ndimage.label(~open_col)
        if trace:
            line_trace.append({"line": li, "locals": [r["k"] for r in g], "c": c.tolist(), "n": n.tolist(), "lo": float(lo),
                               "hi": float(hi), "cell": CELL, "z_lo": Z_LO,
                               "code": "".join(np.where(open_col, "O", np.where(sup, "#", np.where(hard.sum(1) > 0, "+",
                                                         np.where(free.sum(1) > 0, "o", ".")))).tolist()),
                               "solid": (solid >= MIN_RAYS).astype(int).tolist(), "free": free.astype(int).tolist()})
        if dump:
            code = np.where(open_col, "O", np.where(sup, "#", np.where(hard.sum(1) > 0, "+", np.where(free.sum(1) > 0, "o", "."))))
            print(f"  line {li}: facing {np.degrees(np.arctan2(n[1], n[0])):+.0f} deg {''.join(code)}")
        for k in range(1, runs.max() + 1):
            cols = np.flatnonzero(runs == k)
            sel = idx[(al >= lo + cols[0] * CELL) & (al < lo + (cols[-1] + 1) * CELL)]
            if len(sel) * VOXEL ** 2 >= MIN_LOCAL:
                pieces.append({"idx": sel, "line": li})
            elif trace:
                line_trace[-1].setdefault("dropped_small", []).append([float(lo + cols[0] * CELL), float(lo + (cols[-1] + 1) * CELL)])
        ol, _ = ndimage.label(open_col)
        for k in range(1, ol.max() + 1):
            cols = np.flatnonzero(ol == k)
            if cols[0] == 0 or cols[-1] == len(open_col) - 1:
                continue                                  # beyond the ends: not between two pieces
            a0, a1 = lo + cols[0] * CELL, lo + (cols[-1] + 1) * CELL
            cuts.append({"line": li, "p0": (c + a0 * u).tolist(), "p1": (c + a1 * u).tolist(),
                         "length_m": float(a1 - a0), "rays": int(through[cols].sum())})
    print(f"ray test: {len(lines)} lines -> {len(pieces)} pieces, {len(cuts)} cuts")

    # ---- 3d: one wall per piece, with the openings the rays find inside it
    kept, rejected = [], []
    for pc in pieces:
        c, n, idx = robust_vertical(P, Nw, pc["idx"])
        u = np.array([-n[1], n[0]])
        al = (P[idx, :2] - c) @ u
        dist = (P[idx, :2] - c) @ n
        fr, cnt = np.unique(Fw[idx], return_counts=True)
        fr = fr[cnt * VOXEL ** 2 > 0.05]
        rec = {"point": c.tolist(), "normal": n.tolist(), "along": u.tolist(),
               "a": [float(np.percentile(al, 0.5)), float(np.percentile(al, 99.5))],
               "z": [float(np.percentile(P[idx, 2], 0.5)), float(np.percentile(P[idx, 2], 99.5))],
               "area_m2": float(len(idx) * VOXEL ** 2), "frames": [int(x) for x in fr],
               "masks": [int(x) for x in np.unique(Mw[idx])], "rms_cm": float(np.sqrt((dist ** 2).mean()) * 100),
               "heading_deg": float(np.degrees(np.arctan2(n[1], n[0]))), "idx": idx, "line": pc["line"]}
        rec["length_m"] = rec["a"][1] - rec["a"][0]
        solid, through = ray_grid(rays, c, n, rec["a"][0], rec["a"][1])
        free = (through >= MIN_RAYS) & (through > 3 * solid)
        rec["seen_share"] = float(((solid >= MIN_RAYS) | free).mean())
        rec["solid_share"] = float((solid >= MIN_RAYS).mean())
        rec["openings"] = []
        lab_, k_ = ndimage.label(free)
        for k in range(1, k_ + 1):
            ia, iz = np.nonzero(lab_ == k)
            if len(ia) * CELL ** 2 < MIN_OPENING:
                continue
            a0, a1 = rec["a"][0] + ia.min() * CELL, rec["a"][0] + (ia.max() + 1) * CELL
            z0, z1 = Z_LO + iz.min() * CELL, Z_LO + (iz.max() + 1) * CELL
            above = solid[ia.min():ia.max() + 1, iz.max() + 1:]
            bounds = opening_bounds(solid, free, ia.min(), ia.max() + 1, iz.min(), iz.max() + 1)
            seen_bottom = z0
            if bounds["bottom"] == "floor":
                z0 = 0.0                                   # nothing solid under it: the hole reaches the floor
            rec["openings"].append({"a": [float(a0), float(a1)], "z": [float(z0), float(z1)],
                                    "seen_bottom_m": float(seen_bottom), "bounds": bounds,
                                    "width_m": float(a1 - a0), "height_m": float(z1 - z0),
                                    "cells_m2": float(len(ia) * CELL ** 2),
                                    "fill": float(len(ia) * CELL ** 2 / ((a1 - a0) * (z1 - z0))),
                                    "rays": int(through[lab_ == k].sum()),
                                    "closed_above": bool(above.size and (above >= MIN_RAYS).any(1).mean() > 0.5),
                                    "corners": [[*(c + a0 * u), z0], [*(c + a1 * u), z0], [*(c + a1 * u), z1], [*(c + a0 * u), z1]]})
        ok = rec["area_m2"] >= MIN_AREA and len(rec["frames"]) >= MIN_FRAMES
        (kept if ok else rejected).append(rec)
    kept.sort(key=lambda r: -r["area_m2"])
    for i, r in enumerate(kept):
        r["id"] = f"W{i + 1}"
    for i, r in enumerate(rejected):
        r["id"] = f"R{i + 1}"
        r["why"] = "support area below the floor" if r["area_m2"] < MIN_AREA else "seen in one frame only"
    plane_of = np.full(len(P), -1, np.int16)
    for i, r in enumerate(kept + rejected):
        plane_of[r["idx"]] = i
    print(f"kept {len(kept)} walls, rejected {len(rejected)}; {int((horiz & (plane_of < 0)).sum())} wall points unassigned")
    for r in kept + rejected:
        ops = "; ".join(f"opening {o['width_m']:.2f}x{o['height_m']:.2f} m" for o in r["openings"])
        print(f"  {r['id']:3} facing {r['heading_deg']:+7.1f} deg  {r['length_m']:.2f} m  {r['area_m2']:.2f} m2  "
              f"{len(r['frames'])} frames  rms {r['rms_cm']:.1f} cm  {r.get('why', '')} {ops}")

    if trace:
        # each stage as it happened, for pagegen.build_wall_steps; thinned points carry the stage labels
        local_of = np.full(len(P), -1, np.int32)
        for r in local:
            local_of[r["idx"]] = r["k"]
        line_of_local = {k: li for li, g in enumerate(lines) for k in [r["k"] for r in g]}
        np.savez_compressed(d / "walls_trace.npz", P=P.astype(np.float32), N=Nw.astype(np.float32), mask=Mw, frame=Fw,
                            horiz=horiz, local=local_of,
                            line=np.array([line_of_local.get(int(k), -1) if k >= 0 else -1 for k in local_of], np.int32),
                            plane=plane_of)
        (d / "walls_trace.json").write_text(json.dumps({
            "voxel_m": VOXEL, "params": {"tol_m": TOL, "split_deg": SPLIT_DEG, "drift_m": DRIFT, "facing_deg": FACING_DEG,
                                         "min_local_m2": MIN_LOCAL, "min_area_m2": MIN_AREA, "min_frames": MIN_FRAMES,
                                         "cell_m": CELL, "min_rays": MIN_RAYS, "horizontal_deg": HORIZ_DEG},
            "locals": [{"k": r["k"], "c": r["c"].tolist(), "n": r["n"].tolist(), "rep": r["rep"], "sweep": r["from_sweep"],
                        "points": int(len(r["idx"])), "mask": int(np.bincount(Mw[r["idx"]]).argmax())} for r in local],
            "reps_deg": [float(np.degrees(a)) for a in reps], "lines": line_trace,
            "planes": [{"id": r["id"], "line": r["line"]} for r in kept + rejected]}))
        print(f"trace -> {d / 'walls_trace.npz'}, {d / 'walls_trace.json'}")

    np.savez_compressed(d / "geom.npz", X=X.astype(np.float32), N=N.astype(np.float32), cams=C.astype(np.float32),
                        wall_first=np.flatnonzero(w)[first], wall_plane=plane_of, wall_horizontal=horiz)
    strip = lambda r: {k: v for k, v in r.items() if k != "idx"}
    doc = {"run": run,
           "params": {"voxel_m": VOXEL, "voxel_axis_deg": float(np.degrees(axis)), "tol_m": TOL, "split_deg": SPLIT_DEG, "drift_m": DRIFT, "facing_deg": FACING_DEG,
                      "cell_m": CELL, "min_rays": MIN_RAYS, "min_local_m2": MIN_LOCAL, "min_area_m2": MIN_AREA,
                      "min_frames": MIN_FRAMES, "min_opening_m2": MIN_OPENING, "horizontal_deg": HORIZ_DEG},
           "floor": floor, "walls": [strip(r) for r in kept], "rejected": [strip(r) for r in rejected], "cuts": cuts,
           "scene_rays": int(len(rays.P)), "scene_frames": int(rays.frames), "lines": len(lines),
           "wall_points": int(w.sum()), "wall_points_thinned": int(len(P)), "wall_points_horizontal": int(horiz.sum()),
           "wall_points_unassigned": int((horiz & (plane_of < 0)).sum()), "local_planes": len(local)}
    (d / "walls.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--dump", action="store_true", help="print each wall line's ray-test columns")
    ap.add_argument("--trace", action="store_true", help="also write walls_trace.{npz,json}: every stage, for the step page")
    a = ap.parse_args()
    walls(a.run, a.dump, a.trace)


if __name__ == "__main__":
    main()
