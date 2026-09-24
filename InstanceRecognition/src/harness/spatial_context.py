"""Read-only spatial hints from the window's prior committed state."""
import numpy as np
from scipy.spatial import cKDTree
from src.visualization.object_bboxes import export_version_objects


def build_spatial_context(data, state, observations):
    scene = {'base_version': state['version'], 'cutoff': state['cutoff'],
             'coordinate_frame': 'shared_world', 'units': 'meters',
             'geometry_scope': 'archived observed surfaces, not complete physical extents',
             'bbox_convention': 'upright OBB; local Z equals world Z; axes_world are local axes expressed in world coordinates',
             'objects': []}
    candidates = {o['id']: [] for o in observations}
    if not state['objects']:
        return scene, candidates
    version_dir = data.out/'versions'/f'{state["version"]:04d}'
    # Missing committed files are corruption: fail rather than invent geometry.
    exported = export_version_objects(version_dir, state, data.out)
    trees = {}
    for item in exported:
        oid = item['object_id']
        obb = item['bbox_3d_obb']
        scene['objects'].append({'object_id': oid, 'object_label': item['object_label'],
            'bbox_3d_obb': {k: obb[k] for k in ('center_xyz','size_xyz','axes_world')} if obb else None,
            'point_count': item['point_count']})
        with np.load(version_dir/'clouds'/f'{oid}.npz', allow_pickle=False) as z:
            points = z['points']
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            trees[oid] = cKDTree(points)
    for observation in observations:
        points = data.points[observation['id']]
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        scores = [(float(np.median(tree.query(points, k=1)[0])), oid) for oid, tree in trees.items()]
        scores.sort(key=lambda pair: (pair[0], int(pair[1].removeprefix('Object'))))
        candidates[observation['id']] = [
            {'rank': rank, 'object_id': oid, 'object_label': state['objects'][oid]['object_label'],
             'median_surface_distance_m': distance}
            for rank, (distance, oid) in enumerate(scores[:3], 1)]
    return scene, candidates
