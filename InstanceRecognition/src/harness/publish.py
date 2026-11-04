"""Materialize committed windows.

This module writes PIPELINE output only. Everything the browser displays lives in
`pagegen/`; see `pagegen/build_site.py` for the page side.
"""
import os
from pathlib import Path
import numpy as np

from src.harness.state import save


def materialize(data, checked, version_dir, previous, voxel):
    """Write one committed version's per-object clouds, state and transaction."""
    state = checked['state']
    (version_dir / 'clouds').mkdir(parents=True, exist_ok=True)
    for oid in state['objects']:
        target = version_dir / 'clouds' / f'{oid}.npz'
        if oid not in checked['affected'] and previous and (previous / 'clouds' / f'{oid}.npz').exists():
            os.link(previous / 'clouds' / f'{oid}.npz', target)
        else:
            p, c = data.fused(state, oid, voxel)
            np.savez_compressed(target, points=p, rgb=c)
    save(version_dir / 'transaction.json', checked)
    save(version_dir / 'state.json', state)
