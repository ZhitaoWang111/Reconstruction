#!/usr/bin/env python3
"""Run official AMB3R-VO and export full point maps without auxiliary sky models.

The downstream evaluator and mask projection consume the unfiltered point maps.
The optional upstream sky/edge-filtered keyframe preview is not generated here.
"""
import argparse
from functools import partial
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import runtime_compat
from run_headless_amb3r import write_binary_ply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--demo-name', default='scene0000_00_uniform500')
    parser.add_argument('--sdpa-attention', action='store_true')
    parser.add_argument('--interpolation-chunk-size', type=int, default=8192,
                        help='Point interpolation work batch; does not subsample geometry')
    args = parser.parse_args()
    if args.interpolation_chunk_size <= 0:
        parser.error('--interpolation-chunk-size must be positive')
    for name in ['repo', 'data', 'checkpoint', 'output']:
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    prediction = args.output / f'scene_{args.demo_name}_results.npz'
    if prediction.exists():
        raise FileExistsError(prediction)
    sys.path[:0] = [str(args.repo), str(args.repo / 'thirdparty')]
    os.chdir(args.repo)
    if args.sdpa_attention:
        runtime_compat.enable_sdpa_attention()
    # evo imports a plotting backend even though VO does not request plots.
    from evo.tools.settings import SETTINGS
    SETTINGS.plot_backend = 'Agg'
    from amb3r.model_zoo import load_model
    from slam.pipeline import AMB3R_VO
    from slam.datasets.demo import DemoDataset

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model('amb3r', ckpt_path=str(args.checkpoint)).cuda().eval()
    model.backend.forward = partial(model.backend.forward, chunk_size=args.interpolation_chunk_size)
    print(f'Backend interpolation chunk: {args.interpolation_chunk_size} points', flush=True)
    pipeline = AMB3R_VO(model)
    data = DemoDataset(ROOT=str(args.data), resolution=(518, 392))
    _, views = next(iter(DataLoader(data, batch_size=1, shuffle=False, num_workers=0)))
    images = views['images']
    print(f'Processing {images.shape[1]} RGB frames', flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = pipeline.run(images)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    points = result.pts.cpu().numpy()
    confidence = result.conf.cpu().numpy()
    poses = result.poses.cpu().numpy()
    rgb = images.squeeze(0).numpy()
    assert np.isfinite(points).all() and np.isfinite(poses).all() and np.isfinite(confidence).all()
    temporary = prediction.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, pts=points, conf=confidence, pose=poses,
                        images=rgb, kf_idx=result.kf_idx.cpu().numpy(),
                        fps=images.shape[1] / elapsed)
    temporary.replace(prediction)
    # Bounded raw preview solely for the existing metric-scale script's PLY I/O.
    # Canonical geometry is sampled independently by the original evaluator.
    parts, colors = [], []
    rng = np.random.default_rng(0)
    for pointmap, image in zip(points, rgb):
        flat = pointmap.reshape(-1, 3)
        take = rng.choice(len(flat), min(1000, len(flat)), replace=False)
        parts.append(flat[take])
        colors.append(((image.transpose(1, 2, 0).reshape(-1, 3) + 1) / 2)[take])
    write_binary_ply(args.output / 'raw_preview_rgb.ply', np.concatenate(parts), np.concatenate(colors))
    report = dict(frames=len(points), inference_seconds=elapsed,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  checkpoint=str(args.checkpoint), prediction=str(prediction),
                  sdpa_attention=args.sdpa_attention, sdpa_calls=runtime_compat.SDPA_CALLS,
                  interpolation_chunk_size=args.interpolation_chunk_size,
                  geometry='official AMB3R_VO full point maps; no sky/edge filtering',
                  gt_used_for_inference=False)
    (args.output / 'run_metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
