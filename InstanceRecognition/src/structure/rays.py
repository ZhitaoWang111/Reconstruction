"""Step 1b: the whole scene as rays, for the ray test (walls.py, close.py).

    python -m src.structure.rays

Every 4th pixel of every AMB3R point map of the sequence - all frames, not only the
segmented ones, and not only masked pixels: the 3D point, the camera it was seen from,
and the pixel's colour (for display). The space along a ray from the camera to its
point is empty. No mask, no label, no annotation.

Points are in the lift frame, like lift.py's. Writes artifacts/structure/scene_rays.npz.
"""
from __future__ import annotations

import numpy as np

from src.geometry.amb3r_mask_source import _sample_rgb

from .frames import FRAMES, OUT_ROOT, RAYS, PointMaps, frame_slots

STRIDE = 4


def rays(maps: PointMaps | None = None) -> int:
    maps = maps or PointMaps()
    grid = (np.arange(maps.H)[::STRIDE, None] * maps.W + np.arange(maps.W)[None, ::STRIDE]).reshape(-1)
    P, rgb, cam_of, cams, fids = [], [], [], [], []
    for k, (fid, seq) in enumerate(sorted(frame_slots().items())):
        p = maps.lift(seq)
        ok_all = maps.valid(seq, p)
        ok = ok_all[grid]
        rgb.append(_sample_rgb(FRAMES / f"frame_{fid:06d}.jpg", grid[ok]))
        P.append(p[grid][ok].astype(np.float32)); cam_of.append(np.full(int(ok.sum()), k, np.int16))
        cams.append(maps.camera_lift(seq)); fids.append(fid)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    P = np.concatenate(P)
    np.savez_compressed(RAYS, P=P, rgb=np.concatenate(rgb), cam_of=np.concatenate(cam_of),
                        cams=np.array(cams), frames=np.array(fids))
    print(f"{len(P)} rays from {len(cams)} frames -> {RAYS}")
    return len(P)


def main() -> None:
    rays()


if __name__ == "__main__":
    main()
