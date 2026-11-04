"""Shared observed-support bbox fitting and per-version Harness export.

Only derived files are written. Each window is fitted from its own committed
version cloud; later assignments or points never enter earlier windows.
"""
import hashlib
import json
from pathlib import Path
import scipy
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def surface_planes(points):
    """Extract broad upright surfaces independently of exterior-face eligibility.

    Plane patches provide normals and offsets, never crop tangential extents.
    All thresholds apply to the filtered observed cloud, not semantic truth.
    """
    if len(points) < 120:
        return []
    rng = np.random.default_rng(240924)
    sample = points[rng.choice(len(points), min(3500, len(points)), replace=False)]
    remaining = sample.copy()
    planes = []
    tolerance = 0.02
    for _ in range(6):
        if len(remaining) < 80:
            break
        best = None
        for _ in range(180):
            a, b, c = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(b-a, c-a)
            length = np.linalg.norm(normal)
            if length < 1e-8:
                continue
            normal /= length
            mask = np.abs((remaining-a) @ normal) <= tolerance
            count = int(mask.sum())
            if best is None or count > best[0]:
                best = (count, mask)
        if best is None or best[0] < 80:
            break
        patch = remaining[best[1]]
        center = patch.mean(axis=0)
        _, vectors = np.linalg.eigh((patch-center).T @ (patch-center))
        normal = vectors[:, 0]
        # Refit against full cloud so extraction order does not bias the offset.
        for _ in range(2):
            mask = np.abs((points-center) @ normal) <= tolerance
            patch = points[mask]
            center = patch.mean(axis=0)
            _, vectors = np.linalg.eigh((patch-center).T @ (patch-center))
            normal = vectors[:, 0]
        remaining = remaining[np.abs((remaining-center) @ normal) > tolerance]
        signed = (points-center) @ normal
        outside = min(float(np.mean(signed > tolerance)), float(np.mean(signed < -tolerance)))
        tangent = (patch-center) @ vectors[:, 1:]
        spans = np.diff(np.quantile(tangent, [0.05, 0.95], axis=0), axis=0)[0]
        if len(patch) < max(120, 0.08*len(points)) or min(spans) < 0.08 or np.prod(spans) < 0.02:
            continue
        if np.mean(signed > tolerance) < np.mean(signed < -tolerance):
            normal = -normal  # Positive side points into the object.
        vertical = abs(normal[2]) <= np.sin(np.radians(10))
        horizontal = abs(normal[2]) >= np.cos(np.radians(10))
        if not (vertical or horizontal):
            continue
        upright_normal = normal.copy() if vertical else np.array([0., 0., np.sign(normal[2])])
        if vertical:
            upright_normal[2] = 0.
            upright_normal /= np.linalg.norm(upright_normal)
        projected = points @ upright_normal
        offset = float(np.median(patch @ upright_normal))
        upright_outside = min(float(np.mean(projected < offset-0.025)),
                              float(np.mean(projected > offset+0.025)))
        planes.append({'normal': normal, 'center': center, 'points': patch,
                       'kind': 'vertical' if vertical else 'horizontal',
                       'support_points': len(patch), 'outside_fraction': outside,
                       'upright_outside_fraction': upright_outside,
                       'boundary_eligible': outside <= 0.02 and upright_outside <= 0.02,
                       'score': float(len(patch)*np.prod(spans))})
    return sorted(planes, key=lambda p: p['score'], reverse=True)


def exterior_planes(points):
    """Compatibility entry point: only surfaces eligible to anchor box faces."""
    return [p for p in surface_planes(points) if p["boundary_eligible"]]


def fit_plane_faces(points, axes, origin, lower, upper, planes):
    """Align eligible boundary faces, preserving 98% axis coverage as a guard."""
    records = []
    local = (points-origin) @ axes.T
    occupied = set()
    for plane in planes:
        if not plane['boundary_eligible']:
            continue
        dots = axes @ plane['normal']
        axis = int(np.argmax(np.abs(dots)))
        if abs(dots[axis]) < np.cos(np.radians(10)):
            continue
        values = (plane['points']-origin) @ axes[axis]
        offset = float(np.median(values))
        # Recheck one-sidedness AFTER snapping to upright box axes.
        lo_out = float(np.mean(local[:, axis] < offset-0.025))
        hi_out = float(np.mean(local[:, axis] > offset+0.025))
        side = 'lower' if lo_out <= hi_out else 'upper'
        outside = min(lo_out, hi_out)
        if outside > 0.02 or (axis, side) in occupied:
            continue
        occupied.add((axis, side))
        qlo, qhi = np.quantile(local[:, axis], [0.01, 0.99])
        before = float(lower[axis] if side == 'lower' else upper[axis])
        # Include surface thickness/noise; never use patch width/height to crop.
        if side == 'lower':
            boundary = min(float(np.quantile(values, 0.01))-0.005, float(qlo))
            lower[axis] = boundary
        else:
            boundary = max(float(np.quantile(values, 0.99))+0.005, float(qhi))
            upper[axis] = boundary
        records.append({'axis': axis, 'side': side, 'plane_normal_world': plane['normal'].tolist(),
                        'plane_center_world': plane['center'].tolist(),
                        'support_points': plane['support_points'], 'outside_fraction': outside,
                        'plane_offset_local_m': offset, 'previous_boundary_m': before,
                        'boundary_m': boundary, 'surface_margin_m': 0.005})
    return records



def horizontal_planar_footprint(points, max_samples=3500, k_neighbors=24):
    """Estimate a horizontal XY support footprint from a stable planar patch.

    This is used only for derived bbox fitting. It rejects vertical panels and
    weak patches, then returns robust local-XY bounds plus a support score.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 120:
        return None
    step = max(1, len(points) // max_samples)
    sample = points[::step][:max_samples]
    tree = cKDTree(points)
    _, neighbors = tree.query(sample, k=min(k_neighbors, len(points)))
    best = None
    for idxs in neighbors:
        local = points[np.asarray(idxs)]
        center = local.mean(axis=0)
        cov = (local - center).T @ (local - center) / max(len(local), 1)
        values, vectors = np.linalg.eigh(cov)
        if values[-1] <= 1e-10 or values[0] / values[-1] > 0.12:
            continue
        normal = vectors[:, 0]
        if abs(float(normal[2])) < 0.85:
            continue
        signed = (points - center) @ normal
        support = points[np.abs(signed) <= 0.025]
        if len(support) < max(120, int(0.025 * len(points))):
            continue
        q = support[:, :2] - np.median(support[:, :2], axis=0)
        values_xy = np.linalg.norm(q, axis=1)
        # Robust footprint in the world XY plane; avoid a few projected tails.
        lo, hi = np.quantile(q, [0.01, 0.99], axis=0)
        size = hi - lo
        area = float(np.prod(size))
        if np.any(size < 0.08) or area < 0.04:
            continue
        score = float(len(support)) * area * (1.0 - min(values[0] / values[-1], 0.12) / 0.12)
        candidate = (score, lo + np.median(support[:, :2], axis=0), hi + np.median(support[:, :2], axis=0), len(support), float(area))
        if best is None or score > best[0]:
            best = candidate
    return best

def apply_planar_footprint_filter(points, margin_m=0.025):
    """Keep points inside a supported horizontal plane footprint plus margin."""
    footprint = horizontal_planar_footprint(points)
    if footprint is None:
        return points, {"method": "disabled", "reason": "no_horizontal_planar_support"}
    _, lo, hi, support_points, area = footprint
    keep = np.all((points[:, :2] >= lo - margin_m) & (points[:, :2] <= hi + margin_m), axis=1)
    retained = points[keep]
    fraction = float(len(retained) / len(points)) if len(points) else 0.0
    # A footprint may be a partial shelf or tabletop. Reject it when it would
    # remove too much of the observed object, preserving the old fallback.
    if fraction < 0.65:
        return points, {"method": "rejected", "reason": "retained_fraction_below_guard",
                        "retained_fraction": fraction, "support_points": support_points,
                        "footprint_area_m2": area, "margin_m": margin_m}
    return retained, {"method": "horizontal_planar_footprint", "margin_m": margin_m,
        "support_points": support_points, "footprint_area_m2": area,
        "input_points": int(len(points)), "retained_points": int(len(retained)),
        "retained_fraction": fraction, "footprint_min_xy": lo.tolist(),
        "footprint_max_xy": hi.tolist()}

def supported_quantile_bounds(local, axis, bins=32):
    """Inspect each weak local-axis tail; preserve spatially connected structure.

    Density proposes a boundary, never proves that a point is an outlier.
    Only weak tails without a coherent connection to the interior are removed.
    All changes affect derived bounds only, not the input point cloud.
    """
    values = local[:, axis]
    lo, hi = np.quantile(values, [0.01, 0.99])
    original = [float(lo), float(hi)]
    notes = {'original_bounds_m': original, 'sides': {}}
    if len(local) < 800 or hi - lo < 0.10:
        notes['status'] = 'insufficient_support_for_tail_review'
        return float(lo), float(hi), notes
    hist, edges = np.histogram(values, np.linspace(lo, hi, bins + 1))
    interior = hist[3:-3]
    positive = interior[interior > 0]
    reference = float(np.median(positive)) if len(positive) else 0.0
    threshold = 0.25 * reference
    # Scale connectivity by local sampling distance, within conservative metric limits.
    distances = cKDTree(local).query(local, k=2)[0][:, 1]
    positive_distances = distances[distances > 1e-8]
    radius = float(np.clip(3 * np.median(positive_distances), 0.025, 0.05)) if len(positive_distances) else 0.025
    notes.update(bin_count=bins, reference_bin_count=reference,
                 weak_bin_threshold=threshold, connectivity_radius_m=radius)
    new_bounds = original.copy()
    for side_index, side in enumerate(('lower', 'upper')):
        counts = hist if side == 'lower' else hist[::-1]
        note = {'action': 'keep', 'reason': 'supported_boundary'}
        notes['sides'][side] = note
        if reference == 0 or np.median(counts[:3]) >= threshold:
            continue
        # Walk inward through consecutive weak bins, with at most 1/4 of the span.
        run = 0
        for count in counts[:8]:
            if count >= threshold:
                break
            run += 1
        if run < 3:
            continue
        cut = float(edges[run] if side == 'lower' else edges[-run - 1])
        tail_mask = values < cut if side == 'lower' else values > cut
        tail, core = local[tail_mask], local[~tail_mask]
        note.update(candidate_boundary_m=cut, tail_points=int(len(tail)),
                    tail_fraction=float(len(tail) / len(local)))
        if not len(tail) or not len(core) or len(tail) > 0.08 * len(local):
            note['reason'] = 'tail_too_large_or_empty'
            continue
        pairs = cKDTree(tail).query_pairs(radius, output_type='ndarray')
        graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(tail), len(tail)))
        _, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        dominant = tail[labels == int(np.argmax(sizes))]
        gap = float(cKDTree(core).query(dominant, k=1)[0].min())
        coherent_fraction = float(len(dominant) / len(tail))
        note.update(largest_tail_component_fraction=coherent_fraction,
                    nearest_core_gap_m=gap)
        if coherent_fraction >= 0.60 and gap <= radius:
            note['reason'] = 'continuous_sparse_structure_preserved'
            continue
        note.update(action='retract', reason='weak_tail_without_coherent_connection',
                    retraction_m=abs(cut - original[side_index]))
        new_bounds[side_index] = cut
    notes['final_bounds_m'] = new_bounds
    return new_bounds[0], new_bounds[1], notes


def oriented_bounds(points):
    """Fit a robust upright OBB from plane/hull heading and supported extents.

    World Z is fixed. A broad vertical surface can fix the horizontal heading;
    only an independently verified exterior surface can anchor a boundary. Then
    each final local axis then gets a quantile extent with an additional
    low-density-tail check.  The check acts on the final OBB axes, so it can
    correct a shifted box without changing the proven plane orientation.
    """
    if not len(points):
        return None
    points = np.asarray(points, dtype=np.float64)
    axes = np.eye(3, dtype=np.float64)
    xy = points[:, :2]
    med = np.median(xy, axis=0)
    radial = np.linalg.norm(xy - med, axis=1)
    support = points[radial <= np.quantile(radial, 0.98)]
    candidates = []
    planes = surface_planes(points)
    # Preserve established exterior headings when available. Otherwise retain
    # broad surface orientation, preferring the most one-sided evidence rather
    # than discarding it at the boundary's hard 2% threshold. Interior surfaces
    # may provide orientation, but never gain permission to move a box face.
    verticals = [p for p in planes if p['kind'] == 'vertical']
    exterior = [p for p in verticals if p['boundary_eligible']]
    vertical = exterior[0] if exterior else min(
        verticals, key=lambda p: (max(p['outside_fraction'], p['upright_outside_fraction']),
                                 -p['score']), default=None)
    heading_plane = None if vertical is None else {
        'normal_world': vertical['normal'].tolist(),
        'support_points': vertical['support_points'],
        'outside_fraction': vertical['outside_fraction'],
        'upright_outside_fraction': vertical['upright_outside_fraction'],
        'boundary_eligible': vertical['boundary_eligible'],
        'selection': 'exterior_surface' if exterior else 'broad_surface_most_one_sided',
    }
    plane_angle = float(np.arctan2(vertical["normal"][1], vertical["normal"][0]) % np.pi) if vertical is not None else None
    if plane_angle is not None:
        candidates.append(plane_angle)
    if plane_angle is None and len(support) >= 8:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(support[:, :2])
            hp = support[hull.vertices, :2]
            for a, b in zip(hp, np.roll(hp, -1, axis=0)):
                d = b - a
                if np.linalg.norm(d) < 1e-9:
                    continue
                angle = float(np.arctan2(d[1], d[0]) % np.pi)
                if not any(abs(((angle - old + np.pi / 2) % np.pi) - np.pi / 2) < 1e-4 for old in candidates):
                    candidates.append(angle)
        except Exception:
            pass
    if not candidates:
        centered = xy - xy.mean(axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered / max(len(xy), 1))
        candidates = [float(np.arctan2(vectors[1, -1], vectors[0, -1]) % np.pi)]
        method = 'world_z_xy_pca_fallback'
    else:
        method = ('world_z_exterior_plane_faces' if exterior else 'world_z_surface_plane_heading') if plane_angle is not None else 'world_z_xy_robust_hull_edges'
    best = None
    for angle in candidates:
        u2 = np.array([np.cos(angle), np.sin(angle)])
        v2 = np.array([-u2[1], u2[0]])
        proj = (xy - med) @ np.stack([u2, v2], axis=1)
        bounds = [(float(np.quantile(proj[:, i], 0.01)), float(np.quantile(proj[:, i], 0.99))) for i in range(2)]
        lower_xy = np.array([b[0] for b in bounds]); upper_xy = np.array([b[1] for b in bounds])
        size_xy = upper_xy - lower_xy
        if np.any(size_xy <= 1e-8): continue
        area = float(np.prod(size_xy))
        plane_bonus = 0.0  # Reliable exterior normal fixes yaw, not a score bonus.
        boundary_tol = max(0.015, 0.02 * float(np.linalg.norm(size_xy)))
        support_score = float(np.mean(np.minimum(
            np.minimum(np.abs(proj[:, 0] - lower_xy[0]), np.abs(proj[:, 0] - upper_xy[0])),
            np.minimum(np.abs(proj[:, 1] - lower_xy[1]), np.abs(proj[:, 1] - upper_xy[1]))) < boundary_tol))
        score = area - 0.01 * support_score - plane_bonus
        if best is None or score < best[0]:
            best = (score, angle, lower_xy, upper_xy, area, bounds)
    if best is None:
        center = np.median(points, axis=0)
        return {'center_xyz': center.tolist(), 'size_xyz': [0., 0., 0.],
                'axes_world': axes.tolist(), 'corners_xyz': [center.tolist()] * 8,
                'method': 'world_z_degenerate', 'plane_heading_used': False}
    _, angle, lower_xy, upper_xy, area, xy_notes = best
    u = np.array([np.cos(angle), np.sin(angle), 0.0]); v = np.array([-np.sin(angle), np.cos(angle), 0.0])
    axes[0], axes[1] = u, v
    local = (points - np.array([med[0], med[1], 0.0])) @ axes.T
    reviewed = [supported_quantile_bounds(local, i) for i in range(3)]
    lower_xy = np.array([reviewed[0][0], reviewed[1][0]])
    upper_xy = np.array([reviewed[0][1], reviewed[1][1]])
    z_lower, z_upper, z_notes = reviewed[2]
    lower = np.array([lower_xy[0], lower_xy[1], z_lower]); upper = np.array([upper_xy[0], upper_xy[1], z_upper])
    plane_faces = fit_plane_faces(points, axes, np.array([med[0], med[1], 0.0]), lower, upper, planes)
    center_local = (lower + upper) * 0.5; size = upper - lower
    # Plane snapping can change height bounds; use the final local midpoint.
    origin = np.array([med[0], med[1], 0.0])
    center = origin + center_local @ axes
    signs = np.array([[1 if i & (1 << j) else -1 for j in range(3)] for i in range(8)])
    corners = center + (signs * size * 0.5) @ axes
    projected_corners = (corners - origin) @ axes.T
    assert np.allclose(projected_corners.min(axis=0), lower, atol=1e-9)
    assert np.allclose(projected_corners.max(axis=0), upper, atol=1e-9)
    return {'center_xyz': center.tolist(), 'size_xyz': size.tolist(), 'axes_world': axes.tolist(), 'corners_xyz': corners.tolist(), 'method': method, 'heading_deg': float(np.degrees(angle) % 180), 'fit_area_xy': float(np.prod(size[:2])), 'fit_quantiles': [0.01, 0.99], 'boundary_support': {'axis_u': reviewed[0][2], 'axis_v': reviewed[1][2], 'axis_z': z_notes}, 'plane_heading_used': plane_angle is not None, 'heading_plane': heading_plane, 'plane_faces': plane_faces}


def keep_dominant_component(points, radius_m=0.08, min_fraction=0.75, min_gap_m=0.05):
    """Keep the dominant spatial component when smaller components are detached outliers.

    This is deliberately conservative: all components remain when the cloud is not
    clearly dominated by one separated component, avoiding category-specific priors.
    """
    if len(points) < 20:
        return points, {'method': 'skipped_small_cloud', 'components': 1, 'retained_components': 1}
    pairs = cKDTree(points).query_pairs(radius_m)
    if not pairs:
        return points, {'method': 'no_edges', 'components': len(points), 'retained_components': len(points)}
    rows, cols = [], []
    for i, j in pairs:
        rows.extend((i, j)); cols.extend((j, i))
    graph = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(points), len(points)))
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    dominant = int(order[0])
    dominant_points = points[labels == dominant]
    if len(order) == 1:
        return points, {'method': 'single_component', 'components': 1, 'retained_components': 1}
    rest = points[labels != dominant]
    gap = float(cKDTree(dominant_points).query(rest, k=1)[0].min()) if len(rest) else 0.0
    fraction = float(len(dominant_points) / len(points))
    if fraction >= min_fraction and gap >= min_gap_m:
        return dominant_points, {'method': 'dominant_component', 'radius_m': radius_m,
            'min_fraction': min_fraction, 'min_gap_m': min_gap_m, 'components': int(count),
            'retained_components': 1, 'dominant_points': int(len(dominant_points)), 'removed_component_points': int(len(rest)), 'nearest_removed_gap_m': gap}
    return points, {'method': 'ambiguous_components_kept', 'radius_m': radius_m,
        'min_fraction': min_fraction, 'min_gap_m': min_gap_m, 'components': int(count),
        'retained_components': int(count), 'dominant_fraction': fraction, 'nearest_component_gap_m': gap}


def trim_weak_boundary_points(points, min_points=800, trim_fraction=0.0025):
    """Exclude unsupported extreme tails before fitting the bbox.

    A mask backprojection can create a dense but geometrically thin tail that
    remains connected to the main cloud.  For sufficiently large clouds, use
    a small per-axis quantile trim so one-sided tails cannot set the box extent.
    The source cloud is never changed; this only controls derived bbox fitting.
    """
    if len(points) < min_points:
        return points, {'method': 'skipped_small_cloud', 'trim_fraction': trim_fraction}
    q = np.quantile(points, [trim_fraction, 1.0 - trim_fraction], axis=0)
    keep = np.all((points >= q[0]) & (points <= q[1]), axis=1)
    trimmed = points[keep]
    # Never reduce a cloud below a useful support set.
    if len(trimmed) < max(100, int(0.8 * len(points))):
        return points, {'method': 'trim_rejected', 'trim_fraction': trim_fraction,
                        'input_points': int(len(points)), 'retained_points': int(len(points))}
    return trimmed, {'method': 'axis_quantile_trim', 'trim_fraction': trim_fraction,
                     'input_points': int(len(points)), 'retained_points': int(len(trimmed)),
                     'removed_points': int(len(points) - len(trimmed)),
                     'quantile_min_xyz': q[0].tolist(), 'quantile_max_xyz': q[1].tolist()}

def export_version_objects(version_dir, state=None, source_root=None):
    version_dir = Path(version_dir).resolve()
    # source_root anchors the provenance paths. Under the refactor the versions
    # tree and the site tree are separate, so a caller that knows the records root
    # should pass it rather than relying on the depth-based fallback.
    source_root = Path(source_root).resolve() if source_root else version_dir.parents[1]
    if state is None:
        state = json.loads((version_dir / 'state.json').read_text())
    objects = []
    def bounds(points):
        return {'min_xyz': points.min(axis=0).tolist(), 'max_xyz': points.max(axis=0).tolist()} if len(points) else None

    for oid, obj in sorted(state['objects'].items(), key=lambda item: int(item[0].removeprefix('Object'))):
        source = version_dir / 'clouds' / f'{oid}.npz'
        with np.load(source, allow_pickle=False) as cloud:
            points = cloud['points']
        assert points.ndim == 2 and points.shape[1] == 3
        valid = points[np.isfinite(points).all(axis=1)]
        # Remove sparse statistical outliers only for this derived bbox export.
        # Never rewrite the committed cloud or use category-dependent size priors.
        keep = np.ones(len(valid), dtype=bool)
        threshold = None
        k = min(16, len(valid) - 1)
        if len(valid) >= 20:
            distances = cKDTree(valid).query(valid, k=k + 1)[0][:, 1:].mean(axis=1)
            threshold = float(distances.mean() + 2 * distances.std())
            keep = distances <= threshold
        filtered = valid[keep]
        filtered, component_filter = keep_dominant_component(filtered)
        filtered, boundary_filter = trim_weak_boundary_points(filtered)
        # A second, robust local-density pass is used only for the derived bbox
        # fit.  The instance cloud remains untouched.  Mean/std thresholds are
        # pulled upward by mixed sparse geometry; median+MAD exposes that tail
        # without assuming a category-specific size.
        bbox_points = filtered
        density_core_filter = {"method": "disabled"}
        if len(filtered) >= 800:
            kd = cKDTree(filtered)
            kn = min(16, len(filtered) - 1)
            nn = kd.query(filtered, k=kn + 1)[0][:, 1:].mean(axis=1)
            med_nn = float(np.median(nn))
            mad_nn = float(np.median(np.abs(nn - med_nn)))
            robust_threshold = med_nn + 4.5 * max(mad_nn, 1e-6)
            core_keep = nn <= robust_threshold
            retained_fraction = float(core_keep.mean())
            # Keep the pass conservative: require a substantial core and avoid
            # changing small or already compact clouds.
            if retained_fraction >= 0.65 and retained_fraction <= 0.97:
                candidate = filtered[core_keep]
                candidate, _ = keep_dominant_component(candidate)
                if len(candidate) >= 0.65 * len(filtered):
                    bbox_points = candidate
                    density_core_filter = {
                        "method": "median_knn_plus_4_5mad_for_bbox_only",
                        "neighbors": kn,
                        "median_knn_m": med_nn,
                        "mad_knn_m": mad_nn,
                        "threshold_m": robust_threshold,
                        "input_points": int(len(filtered)),
                        "retained_points": int(len(candidate)),
                        "retained_fraction": float(len(candidate) / len(filtered)),
                    }
        # A detected plane is a partial surface, not the whole object's silhouette.
        # Use its normal/offset for exterior faces, never its footprint to crop extents.
        planar_footprint_filter = {"method": "disabled", "reason": "partial_plane_does_not_bound_whole_object"}
        raw_box, clean_box = bounds(valid), bounds(bbox_points)
        raw_volume = float(np.prod(np.ptp(valid, axis=0))) if len(valid) else 0
        clean_volume = float(np.prod(np.ptp(bbox_points, axis=0))) if len(filtered) else 0
        item = {
            'object_id': oid, 'object_label': obj['object_label'],
            'bbox_3d': clean_box,
            'bbox_3d_raw': raw_box,
            'bbox_3d_obb': oriented_bounds(bbox_points),
            'filtering': {'method': 'statistical_neighbor_distance_then_dominant_component', 'neighbors': k,
                'std_ratio': 2, 'threshold_m': threshold, 'retained_points': int(len(filtered)),
                'removed_points': int(len(valid) - len(filtered)),
                'neighbor_removed_points': int((~keep).sum()),
                'component_filter': component_filter,
                'bbox_fit_retained_points': int(len(bbox_points)), 'boundary_filter': boundary_filter, 'density_core_filter': density_core_filter, 'planar_footprint_filter': planar_footprint_filter,
                'volume_reduction_fraction': 1 - clean_volume / raw_volume if raw_volume else 0},
            'point_count': len(valid),
            'observation_count': sum(a['object_id'] == oid for a in state['assignments'].values()),
            'source_cloud': str(source.relative_to(source_root)) if source.is_relative_to(source_root) else str(source), 'source_cloud_sha256': sha(source),
            'nonfinite_points_excluded': len(points) - len(valid)
        }
        if not len(valid): item['bbox_unavailable_reason'] = 'no_valid_3d_points'
        objects.append(item)
    return objects


def _cached_version_objects(out, version_dir, state, source_root=None):
    # This cache is only used by visualization export; model spatial context
    # continues using the same uncached fitting function and input evidence.
    key = {'schema': 1, 'code_sha256': sha(Path(__file__)),
           'numpy': np.__version__, 'scipy': scipy.__version__,
           'state': state, 'state_sha256': sha(version_dir / 'state.json'),
           'clouds': {oid: sha(version_dir / 'clouds' / f'{oid}.npz')
                      for oid in state['objects']}}
    cache_path = out / 'visualization_cache' / 'bboxes' / f'{state["version"]:04d}.json'
    try:
        cached = json.loads(cache_path.read_text())
        encoded = json.dumps(cached['objects'], sort_keys=True, allow_nan=False).encode()
        if cached['key'] == key and hashlib.sha256(encoded).hexdigest() == cached['objects_sha256']:
            return cached['objects']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    objects = export_version_objects(version_dir, state=state, source_root=source_root)
    encoded = json.dumps(objects, sort_keys=True, allow_nan=False).encode()
    from src.harness.state import save
    save(cache_path, {'key': key, 'objects': objects,
                     'objects_sha256': hashlib.sha256(encoded).hexdigest()})
    return objects


def export_harness_bboxes(out, run_dir=None, states=None):
    """Write an array of object arrays, indexed by zero-based window_index.

    `out` is the site directory (where the JS and its cache are written);
    `run_dir` is the pipeline-records directory (where versions/ lives). They are
    the same directory only for pre-refactor runs.
    """
    out = Path(out).resolve()
    run_dir = Path(run_dir).resolve() if run_dir is not None else out
    if states is None:
        checkpoint = json.loads((run_dir / 'checkpoint.json').read_text())
        states = [json.loads((run_dir / 'versions' / f'{v:04d}' / 'state.json').read_text())
                  for v in range(1, checkpoint['version'] + 1)]
    windows = []
    provenance = []
    for index, state in enumerate(states):
        version = index + 1
        if state['version'] != version:
            raise ValueError(f'Window {index} has unexpected version {state["version"]}')
        version_dir = run_dir / 'versions' / f'{version:04d}'
        objects = _cached_version_objects(out, version_dir, state, source_root=run_dir)
        windows.append(objects)
        provenance.append({'window_index': index, 'state_version': version,
                           'cutoff_seq': state['cutoff'],
                           'source_state': f'versions/{version:04d}/state.json',
                           'source_state_sha256': sha(version_dir / 'state.json')})
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    by_numeric_id = [{str(int(item['object_id'].removeprefix('Object'))): item['bbox_3d_obb']
                      for item in objects} for objects in windows]
    (out / 'object_bboxes.js').write_text(
        'window.HARNESS_OBJECT_BBOXES=' + encode(windows) + ';\n' +
        'window.HARNESS_OBJECT_BBOX_PROVENANCE=' + encode(provenance) + ';\n' +
        'window.HARNESS_BBOX_WINDOWS=' + encode(by_numeric_id) + ';\n')
    return windows
