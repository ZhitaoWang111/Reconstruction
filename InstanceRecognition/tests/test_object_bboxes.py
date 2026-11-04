"""Regression coverage for face snapping and strictly historical bbox sources."""
import hashlib
import json

import numpy as np

from src.visualization import object_bboxes as bbox


def test_horizontal_plane_snapping_uses_final_height_center():
    # A broad bottom plane has real support at z=2. The 5mm surface margin
    # expands only the lower face, so the old pre-snap midpoint is incorrect.
    xy = np.array([(x, y) for x in np.linspace(0, 2, 25)
                   for y in np.linspace(0, 1, 25)])
    bottom = np.column_stack((xy, np.full(len(xy), 2.0)))
    rng = np.random.default_rng(37)
    volume = rng.uniform([0, 0, 2.05], [2, 1, 3.0], size=(800, 3))
    result = bbox.oriented_bounds(np.vstack((bottom, volume)))
    lower_faces = [f for f in result['plane_faces'] if f['axis'] == 2 and f['side'] == 'lower']
    assert lower_faces, 'Synthetic bottom plane must participate in the fit'
    face = lower_faces[0]
    assert face['boundary_m'] < face['previous_boundary_m']
    corners = np.asarray(result['corners_xyz'])
    np.testing.assert_allclose(corners[:, 2].min(), face['boundary_m'], atol=1e-9)
    center = np.asarray(result['center_xyz'])
    size = np.asarray(result['size_xyz'])
    np.testing.assert_allclose(center[2] - size[2] / 2, face['boundary_m'], atol=1e-9)


def test_export_uses_each_window_own_cloud_and_keeps_sources(tmp_path):
    rng = np.random.default_rng(7)
    original = rng.uniform([0, 0, 0], [1, 1, 1], (100, 3))
    for version, offset in [(1, 0), (2, 20)]:
        folder = tmp_path / 'versions' / f'{version:04d}'
        (folder / 'clouds').mkdir(parents=True)
        np.savez(folder / 'clouds' / 'Object1.npz', points=original + offset)
        state = {'version': version, 'cutoff': version * 5 - 1,
                 'objects': {'Object1': {'object_label': f'label_at_{version}'}},
                 'assignments': {f'{version}:001': {'object_id': 'Object1'}}}
        (folder / 'state.json').write_text(json.dumps(state))
    (tmp_path / 'checkpoint.json').write_text(json.dumps({'version': 2}))
    sources = list((tmp_path / 'versions').rglob('*'))
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}
    windows = bbox.export_harness_bboxes(tmp_path)
    assert len(windows) == 2
    for index, items in enumerate(windows):
        item = items[0]
        assert item['object_label'] == f'label_at_{index + 1}'
        assert item['source_cloud'].startswith(f'versions/{index + 1:04d}/')
        assert item['bbox_3d']['min_xyz'][0] >= index * 20
        assert item['bbox_3d']['max_xyz'][0] <= index * 20 + 1
    assert digests == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in digests}
    text = (tmp_path / 'object_bboxes.js').read_text()
    assert text.startswith('window.HARNESS_OBJECT_BBOXES=')
    assert 'window.HARNESS_OBJECT_BBOX_PROVENANCE=' in text
    assert 'window.HARNESS_BBOX_WINDOWS=' in text


def test_broad_interior_surface_guides_heading_without_anchoring_face():
    rng = np.random.default_rng(142)
    # A well-supported panel with observations on both sides is useful for yaw,
    # but must not cut the bounding box through the object's interior.
    panel = np.column_stack((rng.normal(0, .002, 1200),
                             rng.uniform(-.8, .8, 1200),
                             rng.uniform(.1, 1.9, 1200)))
    volume = rng.uniform([-.3, -.8, .1], [.3, .8, 1.9], (800, 3))
    angle = np.radians(28)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    points = np.vstack((panel, volume)) @ rotation.T
    original = points.copy()
    result = bbox.oriented_bounds(points)
    assert result['plane_heading_used']
    assert not result['heading_plane']['boundary_eligible']
    assert result['plane_faces'] == []
    assert result['method'] == 'world_z_surface_plane_heading'
    assert abs(result['heading_deg'] - 28) < 2
    assert result['size_xyz'][0] > .5
    np.testing.assert_allclose(result['axes_world'][2], [0, 0, 1])
    np.testing.assert_array_equal(points, original)


def test_no_broad_plane_keeps_geometric_fallback():
    rng = np.random.default_rng(99)
    points = rng.normal(size=(2000, 3))
    result = bbox.oriented_bounds(points)
    assert not result['plane_heading_used']
    assert result['method'] == 'world_z_xy_robust_hull_edges'
    assert result['plane_faces'] == []


def test_vertical_exterior_surface_still_anchors_face():
    rng = np.random.default_rng(81)
    panel = np.column_stack((np.zeros(1200), rng.uniform(-1, 1, 1200),
                             rng.uniform(0, 2, 1200)))
    volume = rng.uniform([.04, -1, 0], [.7, 1, 2], (800, 3))
    result = bbox.oriented_bounds(np.vstack((panel, volume)))
    assert result['heading_plane']['boundary_eligible']
    assert result['method'] == 'world_z_exterior_plane_faces'
    faces = [f for f in result['plane_faces'] if f['axis'] != 2]
    assert faces
    assert result['size_xyz'][0] > .6
