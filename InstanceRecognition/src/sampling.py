"""Deterministic temporal sampling for ScanNet frame sequences."""

from __future__ import annotations


def uniform_sample_indices(frame_count: int, target_frames: int = 500) -> list[int]:
    """Return up to ``target_frames`` indices spanning the complete sequence."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if frame_count <= target_frames:
        return list(range(frame_count))
    if target_frames == 1:
        return [0]
    last = frame_count - 1
    return [round(index * last / (target_frames - 1)) for index in range(target_frames)]
