"""Derived visualization caches invalidate on source or artifact changes."""
import json
from types import SimpleNamespace
import numpy as np
from src.visualization import object_bboxes as bbox
from pagegen import build_site as publish


def fixture_version(tmp_path):
    folder = tmp_path / 'versions' / '0001'
    (folder / 'clouds').mkdir(parents=True)
    state = {'version': 1, 'cutoff': 4, 'objects': {'Object1': {'object_label': 'desk'}},
             'assignments': {'0000:001': {'object_id': 'Object1'}}}
    (folder / 'state.json').write_text(json.dumps(state))
    (folder / 'transaction.json').write_text('{}')
    np.savez(folder / 'clouds' / 'Object1.npz', points=np.arange(30).reshape(10, 3))
    return folder, state


def test_bbox_cache_reuses_then_invalidates_source_and_corruption(tmp_path, monkeypatch):
    folder, state = fixture_version(tmp_path)
    calls = []
    def export(*args, **kwargs):
        calls.append(1)
        return [{'value': len(calls)}]
    monkeypatch.setattr(bbox, 'export_version_objects', export)
    assert bbox._cached_version_objects(tmp_path, folder, state) == [{'value': 1}]
    assert bbox._cached_version_objects(tmp_path, folder, state) == [{'value': 1}]
    np.savez(folder / 'clouds' / 'Object1.npz', points=np.ones((10, 3)))
    assert bbox._cached_version_objects(tmp_path, folder, state) == [{'value': 2}]
    cache = tmp_path / 'visualization_cache/bboxes/0001.json'
    content = json.loads(cache.read_text()); content['objects'] = []
    cache.write_text(json.dumps(content))
    assert bbox._cached_version_objects(tmp_path, folder, state) == [{'value': 3}]


def test_snapshot_cache_checks_sources_outputs_and_latest_ply(tmp_path):
    folder, state = fixture_version(tmp_path)
    data = SimpleNamespace(items={'0000:001': {'seq': 0}})
    key = publish._snapshot_cache_key(data, folder, state)
    artifact = tmp_path / 'snapshot.js'; artifact.write_text('data')
    cloud = tmp_path / 'object.ply'; cloud.write_text('cloud')
    cache = tmp_path / 'cache.json'
    cache.write_text(json.dumps({'key': key, 'latest_ply_exported': True, 'artifacts': {'snapshot.js': publish._file_hash(artifact)},
                                 'ply': {'object.ply': publish._file_hash(cloud)}}))
    assert publish._snapshot_cache_hit(tmp_path, cache, key, True)
    cloud.write_text('changed')
    assert not publish._snapshot_cache_hit(tmp_path, cache, key, True)
    assert publish._snapshot_cache_hit(tmp_path, cache, key, False)
    artifact.unlink()
    assert not publish._snapshot_cache_hit(tmp_path, cache, key, False)
    (folder / 'transaction.json').write_text('{"changed":true}')
    assert publish._snapshot_cache_key(data, folder, state) != key
