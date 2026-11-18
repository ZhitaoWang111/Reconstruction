"""Step 1: lift the kept masks of a structure segmentation run (src.structure.segment) into 3D.

    python -m src.structure.lift --run struct_v4_win5

One mask pixel gives one point: the AMB3R point map at that pixel, in the amb3r_floor frame
(z = 0 on the floor, +z up; see frames.py). Per point: position, surface normal from the
point-map grid (turned towards the camera), colour, frame, mask index and label.

Writes `points.npz` and `masks.json` to artifacts/structure/<run>/.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

from src.geometry.amb3r_mask_source import AMBR3R_INTERMEDIATE, AMBR3R_SIZE, _sample_rgb
from src.geometry.backproject_masks import align_mask

from .frames import FRAMES, PointMaps, frame_slots, masks_dir, run_dir

LABELS = ("wall", "floor", "door", "window")
NORMAL_STEP = 3            # grid cells each side for the normal's finite difference


def grid_normals(P: np.ndarray, cam: np.ndarray, step: int = NORMAL_STEP) -> np.ndarray:
    """Normal per cell of an (H, W, 3) point map, turned towards the camera; NaN where the stencil leaves the grid."""
    n = np.full_like(P, np.nan)
    dx = P[step:-step, 2 * step:] - P[step:-step, :-2 * step]
    dy = P[2 * step:, step:-step] - P[:-2 * step, step:-step]
    c = np.cross(dx, dy)
    c /= np.maximum(np.linalg.norm(c, axis=2, keepdims=True), 1e-12)
    c[((cam - P[step:-step, step:-step]) * c).sum(2) < 0] *= -1
    n[step:-step, step:-step] = c
    return n


def lift(run: str, maps: PointMaps | None = None) -> dict:
    maps = maps or PointMaps()
    slot = frame_slots()
    cols = {k: [] for k in ("xyz", "normal", "rgb", "frame", "mask", "label")}
    cams, masks, dropped = {}, [], 0
    for folder in sorted(masks_dir(run).glob("frame_*")):
        result = json.loads((folder / "result.json").read_text())
        fid = int(result["frame_id"]); seq = slot[fid]
        pts = maps.points(seq)
        cam = maps.camera(seq)
        cams[fid] = cam
        nrm = grid_normals(pts.reshape(maps.H, maps.W, 3), cam).reshape(-1, 3)
        ok_all = maps.valid(seq, pts) & np.isfinite(nrm).all(1)
        for kept in result["kept"]:
            m = align_mask(np.asarray(Image.open(folder / kept["mask_file"]).convert("L")) > 0,
                           AMBR3R_INTERMEDIATE, AMBR3R_SIZE).reshape(-1)
            ok = m & ok_all
            dropped += int((m & ~ok).sum())
            idx = np.flatnonzero(ok)
            mi = len(masks)
            masks.append({"mask": mi, "frame": fid, "plan_id": kept["plan_id"], "label": kept["label"],
                          "where": kept["where"], "pixels": int(m.sum()), "points": int(len(idx))})
            cols["xyz"].append(pts[idx].astype(np.float32)); cols["normal"].append(nrm[idx].astype(np.float32))
            cols["rgb"].append(_sample_rgb(FRAMES / f"frame_{fid:06d}.jpg", idx))
            cols["frame"].append(np.full(len(idx), fid, np.int32)); cols["mask"].append(np.full(len(idx), mi, np.int32))
            cols["label"].append(np.full(len(idx), LABELS.index(kept["label"]), np.uint8))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    d = run_dir(run)
    np.savez_compressed(d / "points.npz", **out, cam_frames=np.array(sorted(cams)),
                        cam_xyz=np.array([cams[f] for f in sorted(cams)]))
    (d / "masks.json").write_text(json.dumps(masks, indent=1, ensure_ascii=False))
    print(f"{len(masks)} masks, {len(out['xyz'])} points ({dropped} mask pixels dropped: low confidence or grid border)")
    for i, label in enumerate(LABELS):
        print(f"  {label:6} {(out['label'] == i).sum():8d} points in {sum(1 for m in masks if m['label'] == label)} masks")
    return {"masks": len(masks), "points": int(len(out["xyz"]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    lift(ap.parse_args().run)


if __name__ == "__main__":
    main()
