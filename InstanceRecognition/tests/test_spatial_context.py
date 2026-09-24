from types import SimpleNamespace
import numpy as np
from src.harness.spatial_context import build_spatial_context


def test_frozen_geometry_and_robust_candidates(tmp_path):
    objects = {f'Object{i}': {'object_label': 'same category'} for i in range(1, 5)}
    state = {'version': 1, 'cutoff': 4, 'objects': objects, 'assignments': {}}
    for version in (1, 2):
        folder = tmp_path/'versions'/f'{version:04d}'/'clouds'
        folder.mkdir(parents=True)
        for i in range(1, 5):
            np.savez(folder/f'Object{i}.npz', points=np.array([[i if version == 1 else 100, 0., 0.]]))
    data = SimpleNamespace(out=tmp_path, points={'a': np.array([[1.,0,0],[1.,0,0],[4.,0,0],[np.nan,0,0]]), 'empty': np.empty((0,3))})
    scene, candidates = build_spatial_context(data, state, [{'id':'a'}, {'id':'empty'}])
    assert [c['object_id'] for c in candidates['a']] == ['Object1','Object2','Object3']
    assert candidates['a'][0]['median_surface_distance_m'] == 0
    assert candidates['empty'] == []
    assert scene['base_version'] == 1 and scene['cutoff'] == 4
    assert scene['objects'][0]['bbox_3d_obb']['center_xyz'][0] == 1


def test_first_window_no_history(tmp_path):
    scene, candidates = build_spatial_context(SimpleNamespace(out=tmp_path),
        {'version':0,'cutoff':-1,'objects':{}}, [{'id':'a'}])
    assert scene['objects'] == [] and candidates == {'a':[]}
