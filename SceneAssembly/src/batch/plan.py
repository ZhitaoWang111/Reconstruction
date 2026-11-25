"""Cutting the scene into batches (D02)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

BATCH_SIZE = 5


@dataclass(frozen=True)
class Batch:
    index: int                 # 1-based
    objects: tuple             # in first-key-frame order
    frames: tuple              # the objects' first key frames, unique, ascending

    def to_dict(self):
        return {'index': self.index, 'objects': list(self.objects), 'frames': list(self.frames)}


def first_key_frame(obj) -> int:
    return min(f.seq for f in obj.key_frames)


def cut_batches(scene, size: int = BATCH_SIZE) -> list:
    order = sorted(scene.objects, key=lambda o: (first_key_frame(scene.objects[o]), o))
    out = []
    for i in range(0, len(order), size):
        oids = tuple(order[i:i + size])
        frames = tuple(sorted({first_key_frame(scene.objects[o]) for o in oids}))
        out.append(Batch(index=len(out) + 1, objects=oids, frames=frames))
    return out


def write_plan(batches, path: Path):
    Path(path).write_text(json.dumps([b.to_dict() for b in batches], indent=1))
