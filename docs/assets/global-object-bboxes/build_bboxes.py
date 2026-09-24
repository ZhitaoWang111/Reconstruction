"""Export first30 world-Z-aligned, robust horizontal-heading boxes, legacy AABBs and scene reference.
Run: InstanceRecognition/.venv/bin/python docs/assets/global-object-bboxes/build_bboxes.py
Only derived viewer data is written; committed experiment records remain unchanged.
"""
import base64
import json
import colorsys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'InstanceRecognition/runtime/experiments/harness_window01_v3'

# Share fitting rules with the persistent Harness publisher.
import sys
sys.path.insert(0, str(ROOT / 'InstanceRecognition'))
from src.visualization.object_bboxes import export_version_objects, oriented_bounds, sha

def main():
    version_dir = RUN / 'versions/0006'
    state_path = version_dir / 'state.json'
    state = json.loads(state_path.read_text())
    assert state['version'] == 6 and state['cutoff'] == 29
    config = json.loads((RUN / 'config.json').read_text())
    objects = export_version_objects(version_dir, state=state, source_root=ROOT)
    # Latest complete instance cloud from the harness result (state 0005).
    # Keep the original world coordinates and object IDs so the viewer can
    # compare assigned points directly against the derived bbox.
    latest_state_cloud = RUN / 'states' / '0005.npz'
    with np.load(latest_state_cloud, allow_pickle=False) as latest:
        latest_points = latest['points_world']
        latest_object_ids = latest['object_ids']
        latest_rgb = latest['rgb']
    finite_latest = np.isfinite(latest_points).all(axis=1)
    latest_points = latest_points[finite_latest]
    latest_object_ids = latest_object_ids[finite_latest]
    latest_rgb = latest_rgb[finite_latest]
    # The state file may not persist display colors. Generate a stable, high-contrast
    # palette by Object ID so every instance cloud remains visually distinguishable.
    object_colors = {}
    for index, (oid, obj) in enumerate(sorted(state['objects'].items(), key=lambda item: int(item[0].removeprefix('Object')))):
        numeric_id = int(oid.removeprefix('Object'))
        stored = obj.get('color')
        if isinstance(stored, list) and len(stored) == 3 and all(isinstance(v, (int, float)) for v in stored) and stored != [220, 220, 220]:
            object_colors[numeric_id] = [int(max(0, min(255, v))) for v in stored]
        else:
            hue = ((index * 137.508 + 150.0) % 360.0) / 360.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.68, 0.96)
            object_colors[numeric_id] = [round(r * 255), round(g * 255), round(b * 255)]
    object_labels = {int(oid.removeprefix('Object')): o.get('object_label', '未命名对象') for oid, o in state['objects'].items()}
    latest_colors = np.asarray([object_colors.get(int(i), [220, 220, 220]) for i in latest_object_ids], dtype=np.uint8)

    source_dir = Path(config['source'])
    manifest = json.loads((source_dir / 'manifest.json').read_text())
    background_path = source_dir / manifest['background_asset']
    background_text = background_path.read_text()
    background = json.loads(background_text[background_text.index('{'):background_text.rfind('}') + 1])
    scene_points = np.frombuffer(base64.b64decode(background['positions_f32_base64']), dtype='<f4').reshape(-1, 3)
    color_key = next(k for k in background if 'color' in k and 'base64' in k)
    scene_colors = np.frombuffer(base64.b64decode(background[color_key]), dtype=np.uint8).reshape(-1, 3)
    ids = np.flatnonzero(np.isfinite(scene_points).all(axis=1))
    # Display every finite source point; no additional viewer downsampling.
    payload = {
        'schema_version': 4, 'scene_id': 'scene0000_00', 'state_version': 6, 'cutoff_seq': 29,
        'coordinate_system': {'frame': 'scannet_scan_world', 'unit': 'meter', 'up_axis': 'z',
            'note': 'Existing GT-corrected AMB3R coordinates; no additional axisAlignment or transform applied.'},
        'scene_cloud': {'points': scene_points[ids].round(5).astype(float).tolist(),
            'colors': scene_colors[ids].tolist(), 'source_count': len(scene_points),
            'display_count': len(ids), 'source': str(background_path.relative_to(ROOT)),
            'source_sha256': sha(background_path),
            'scope': 'Full-video background geometry for context, not restricted to first 30 frames; object bboxes are first30 only.'},
        'instance_cloud': {'points': latest_points.round(5).astype(float).tolist(),
            'object_ids': latest_object_ids.astype(int).tolist(),
            'object_labels': {str(k): v for k, v in sorted(object_labels.items())},
            'object_colors': {str(k): v for k, v in sorted(object_colors.items())},
            'colors': latest_colors.tolist(), 'source_count': int(len(latest_points)),
            'source': str(latest_state_cloud.relative_to(ROOT)),
            'source_sha256': sha(latest_state_cloud),
            'scope': 'Latest complete instance point cloud from all first30 processing windows; points retain harness Object IDs and world coordinates.'},
        'bbox_type': 'AABB', 'default_bbox_type': 'OBB', 'orientation_constraint': 'world_z_fixed_xy_robust_hull', 'extent_semantics': 'observed_points', 'objects': objects,
        'provenance': {
            'source_run': str(RUN.relative_to(ROOT)), 'source_state': str(state_path.relative_to(ROOT)),
            'source_state_sha256': sha(state_path), 'source_config_sha256': sha(RUN / 'config.json'),
            'sampled_sequence_range_inclusive': [0, 29], 'frame_count': 30,
            'geometry_policy': config['geometry_policy'], 'source_voxel_size_m': config['voxel'],
            'bounds_policy': 'bbox_3d_obb: world +Z axis fixed to [0,0,1]; reliable exterior vertical plane normals fix horizontal heading; otherwise robust XY hull edges select compact projected extents; exterior horizontal planes constrain height only; plane faces require >=8% support, broad 2D extent and <=2% points beyond their outer side; snapped face positions include 5mm margin and preserve 1%-99% axis coverage; PCA is fallback only for degenerate clouds; axes_world are row vectors and corners_xyz use bit-index corner order. This is an observed-support box, not a complete-object or minimum-volume box. bbox_3d: min/max after mean 16-neighbor distance <= population mean + 2 standard deviations, then retain a clearly dominant separated spatial component (radius 0.08 m, >=75% of points, >=0.05 m gap); ambiguous components are kept. For clouds with >=800 points, apply a 0.25% world-axis trim, a conservative median KNN distance + 4.5 MAD density pass (at least 65% retained), with no plane-footprint cropping (partial surfaces do not bound whole objects); then fit the OBB in its final local axes using 1%-99% quantiles with per-side low-density-tail checks. After heading selection, inspect 32 bins per local axis for clouds with >=800 points and axis span >=0.10 m. A weak tail requires >=3 consecutive outer bins below 25% of median interior occupancy; inspect at most 8 bins and reject tails over 8% of points. Preserve tails whose largest radius-connected component has >=60% of tail points and connects to the interior (radius=3*median nearest-neighbor distance, clipped to 0.025-0.05 m). Retract other weak tails; original 1%-99% base clipping still applies to all structures. This boundary-support check is recorded per axis and does not modify source clouds. bbox_3d_raw: original finite-point min/max. Derived export only; committed clouds unchanged.',
            'limitations': 'Observed extents, not complete object geometry or GT instance annotations. Statistical filtering can remove valid sparse parts and cannot resolve dense misassignments. Legacy AABB includes empty corners for rotated objects. The robust OBB suppresses extreme tails and uses boundary directions, but an incomplete or semantically wrong instance cloud can still produce an imperfect heading. Existing assignment errors are preserved.'
        }
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    (OUT / 'bboxes.json').write_text(text + '\n')
    (OUT / 'data.js').write_text('window.GLOBAL_OBJECT_BBOXES=' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/') + ';\n')
    print(json.dumps({'objects': len(objects), 'points': sum(o['point_count'] for o in objects),
                      'missing_bounds': sum(o['bbox_3d'] is None for o in objects)}, ensure_ascii=False))

if __name__ == '__main__': main()
