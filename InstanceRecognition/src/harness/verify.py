"""Post-proposal verification: render evidence images for hold / create / merge.

After ``propose_update`` validates a proposal structurally the caller passes the
pending proposal here.  For each hold, create, or merge action this module
produces a rendered image and a short text note the model can read to decide
whether to commit or re-propose.

Back-projection math
--------------------
``camera_to_world`` (4×4) maps camera-frame points to the scan frame.  The
inverse maps scan-frame points into the camera frame; then the 3×3 colour
intrinsic projects them to pixel coordinates.  Points behind the camera
(z ≤ 0) and points outside the image are discarded.

The intrinsic is stored as a 4×4 matrix in the manifest
(``[[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0],[0,0,0,1]]``); only the top-left 3×3
is used here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    pass


# A hold is flagged when fewer than this fraction of the object's projected
# points land inside the new observation's mask. Measured on the first 100
# frames of scene0000_00 (frames 2-10 apart): pairs with the same label and
# centroids under 0.3 m fall below it 15% of the time, while pairs within 1 m
# that are not that pair clear it 13% of the time.
HOLD_HIT_THRESHOLD = 0.2
# Fewer projected points than this and the hit rate means nothing; the check
# says so instead of guessing.
MIN_PROJECTED = 20
# Images attached to one verify message. The model reads each at full
# resolution, and a proposal holds ~40 observations; only the worst are shown.
MAX_VERIFY_IMAGES = 6


# ── projection helpers ────────────────────────────────────────────────────────

def _load_frame_assets(data, seq: int):
    """Return (camera_to_world 4×4, K 3×3, image_wh) for a sequence index."""
    source = getattr(data, 'source', None)
    if source is None:
        return None, None, None
    frame_path = Path(source) / 'frames' / f'{seq:04d}.npz'
    if not frame_path.is_file():
        return None, None, None
    with np.load(frame_path) as z:
        if 'camera_to_world' not in z.files:
            return None, None, None
        c2w = z['camera_to_world'].astype(np.float64)
    manifest_path = Path(source) / 'manifest.json'
    if not manifest_path.is_file():
        return None, None, None
    manifest = json.loads(manifest_path.read_text())
    intr = manifest.get('intrinsic_color')
    if intr is None:
        return None, None, None
    K = np.asarray(intr, dtype=np.float64)[:3, :3]
    # image size from manifest frames list
    frames = manifest.get('frames', [])
    wh = (1296, 968)  # fallback
    for f in frames:
        if int(f.get('sequence_index', -1)) == seq:
            instances = f.get('instances', [])
            if instances:
                break
    return c2w, K, wh


def _project_points(pts: np.ndarray, c2w: np.ndarray, K: np.ndarray,
                    image_wh: tuple[int, int]) -> np.ndarray:
    """Project world points to pixel coords; return (N,2) float, NaN for culled."""
    w, h = image_wh
    w2c = np.linalg.inv(c2w)
    pts_h = np.c_[pts, np.ones(len(pts))]
    cam = (w2c @ pts_h.T).T[:, :3]
    valid_depth = cam[:, 2] > 0.01
    px = np.full((len(pts), 2), np.nan)
    if valid_depth.any():
        proj = (K @ cam[valid_depth].T).T
        uv = proj[:, :2] / proj[:, 2:3]
        in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.flatnonzero(valid_depth)
        px[idx[in_frame]] = uv[in_frame]
    return px


# ── cloud helpers ─────────────────────────────────────────────────────────────

def _held_cloud(data, state: dict, object_id: str, extra_obs: list | None = None,
                exclude: list | None = None) -> np.ndarray:
    """Voxelised (2 cm) point cloud from held observations.

    ``extra_obs`` is a list of observation ids to include that are not yet held
    (e.g. the new observation being held in the pending proposal).
    """
    held = [k for k, a in state['assignments'].items()
            if a.get('object_id') == object_id
            and a.get('status') in ('held', 'assigned')]
    if extra_obs:
        held = list(set(held) | set(extra_obs))
    if exclude:
        held = [k for k in held if k not in set(exclude)]
    points_map = getattr(data, 'points', {})
    parts = [points_map[k] for k in held if k in points_map and len(points_map[k])]
    if not parts:
        return np.empty((0, 3), np.float32)
    pts = np.concatenate(parts)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return np.empty((0, 3), np.float32)
    keys = np.floor(pts / 0.02).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def _mask_hit_rate(data, state, object_id, obs_id):
    """Fraction of the object's held cloud that projects inside obs_id's mask.

    Only points that land in the frame count. Returns None when the frame has
    no pose or too few points project to judge.
    """
    obs_meta = data.items.get(obs_id)
    if obs_meta is None:
        return None
    c2w, K, wh = _load_frame_assets(data, obs_meta['seq'])
    if c2w is None:
        return None
    cloud = _held_cloud(data, state, object_id, exclude=[obs_id])
    if not len(cloud):
        return None
    px = _project_points(cloud[::max(1, len(cloud) // 3000)], c2w, K, wh)
    ok = ~np.isnan(px[:, 0])
    if ok.sum() < MIN_PROJECTED:
        return None
    mask_path = Path(obs_meta.get('mask', ''))
    if not mask_path.is_file():
        return None
    with Image.open(mask_path) as m:
        mask = np.asarray(m.convert('L').resize(wh)) > 0
    u = px[ok, 0].astype(int)
    v = px[ok, 1].astype(int)
    return float(mask[v, u].mean())


# ── rendering ─────────────────────────────────────────────────────────────────

def _render_backprojection(data, state: dict, object_id: str,
                           obs_id: str, extra_obs: list | None = None,
                           exclude: list | None = None) -> Path | None:
    """Render the object's cloud back-projected onto the observation's RGB frame.

    Returns the path to the rendered JPEG, or None if rendering is not possible
    (missing camera_to_world, no points, etc.).
    """
    obs_meta = data.items.get(obs_id)
    if obs_meta is None:
        return None
    seq = obs_meta['seq']
    c2w, K, wh = _load_frame_assets(data, seq)
    if c2w is None:
        return None
    cloud = _held_cloud(data, state, object_id, extra_obs, exclude)
    if not len(cloud):
        return None

    # Open the original RGB frame.
    rgb_path = Path(obs_meta.get('image', ''))
    if not rgb_path.is_file():
        return None
    with Image.open(rgb_path) as img:
        frame = img.convert('RGB').resize(wh)

    # Subsample to keep rendering fast.
    step = max(1, len(cloud) // 3000)
    sub = cloud[::step]
    px = _project_points(sub, c2w, K, wh)
    valid = ~np.isnan(px[:, 0])

    draw = ImageDraw.Draw(frame)
    # Overlay observation mask bbox.
    bbox = obs_meta.get('bbox_xyxy')
    if bbox:
        draw.rectangle(bbox, outline=(255, 200, 0), width=2)

    # Draw projected cloud as small cyan dots.
    for u, v in px[valid]:
        r = 2
        draw.ellipse([u - r, v - r, u + r, v + r], fill=(0, 200, 255))

    out_dir = Path(data.out) / 'verify'
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f'{object_id}_{obs_id.replace(":", "-")}'
    out_path = out_dir / f'{key}_backproj.jpg'
    frame.save(out_path, quality=85, subsampling=0)
    return out_path


# ── overlap check ─────────────────────────────────────────────────────────────

def _overlap_fraction(cloud_a: np.ndarray, cloud_b: np.ndarray,
                      tolerance: float = 0.10) -> float:
    """Fraction of cloud_a points within ``tolerance`` metres of cloud_b."""
    if not len(cloud_a) or not len(cloud_b):
        return 0.0
    from scipy.spatial import cKDTree
    tree = cKDTree(cloud_b)
    step = max(1, len(cloud_a) // 2000)
    dist, _ = tree.query(cloud_a[::step], k=1)
    return float((dist <= tolerance).mean())


# ── public API ────────────────────────────────────────────────────────────────

def build_checks(pending: dict, data, state: dict) -> tuple[list[dict], list[Path]]:
    """Return (checks, image_paths) for the pending proposal.

    Each check is a dict with keys:
      action_index  int
      action        'hold' | 'create' | 'merge'
      object_id     str
      note          str   — human-readable finding
      images        list[str]  — relative paths to rendered images (for audit)
      verify_fail   bool  — True means there is something to look at

    ``image_paths`` is the flat list of Path objects to attach to the message.
    """
    checks: list[dict] = []
    image_paths: list[Path] = []
    actions = pending.get('proposal', {}).get('actions', [])
    proposed_state = pending.get('state', state)

    for idx, action in enumerate(actions):
        kind = action.get('action')
        oid = action.get('object_id')

        if kind == 'hold':
            # The question a hold answers is whether the object, as already
            # held, is what the new mask shows. So the cloud is the object's
            # other held observations, projected into the new one's frame.
            # A new object's first holds have nothing else to compare against.
            obs_ids = action.get('observations', [])
            for obs_id in obs_ids:
                check: dict = {
                    'action_index': idx, 'action': 'hold',
                    'object_id': oid, 'observation_id': obs_id,
                    'images': [], 'verify_fail': False,
                }
                rate = _mask_hit_rate(data, proposed_state, oid, obs_id)
                check['hit_rate'] = rate
                if rate is None:
                    check['note'] = (f'{obs_id}: no comparison (no pose, or the rest '
                                     f'of {oid} does not project into this frame).')
                elif rate >= HOLD_HIT_THRESHOLD:
                    check['note'] = (f'{obs_id}: {rate:.0%} of {oid} projects inside '
                                     f'the mask.')
                else:
                    check['verify_fail'] = True
                    check['note'] = (
                        f'{obs_id}: only {rate:.0%} of {oid} projects inside the mask. '
                        f'Cyan dots are the rest of {oid} drawn into this frame; the '
                        f'yellow box is the mask. If they do not cover the same thing, '
                        f'this observation is not {oid}.')
                checks.append(check)

        elif kind == 'create':
            # The new object's cloud comes from the proposed state's held
            # observations, not from the create action itself (create carries
            # no observations field; the hold action that follows it does).
            new_cloud = _held_cloud(data, proposed_state, oid)
            worst_oid, worst_frac = None, 0.0
            for other_oid in state['objects']:
                if other_oid == oid:
                    continue
                other_cloud = _held_cloud(data, state, other_oid)
                frac = _overlap_fraction(new_cloud, other_cloud)
                if frac > worst_frac:
                    worst_frac, worst_oid = frac, other_oid
            check = {
                'action_index': idx, 'action': 'create',
                'object_id': oid, 'images': [], 'verify_fail': False,
            }
            if worst_frac > 0.10 and worst_oid:
                # Find a held observation of the overlapping object to render on.
                held_obs = [k for k, a in state['assignments'].items()
                            if a.get('object_id') == worst_oid
                            and a.get('status') in ('held', 'assigned')]
                if held_obs:
                    img = _render_backprojection(data, state, worst_oid, held_obs[0])
                    if img:
                        check['images'].append(str(img))
                        image_paths.append(img)
                check['note'] = (
                    f'New {oid} overlaps {worst_oid} by '
                    f'{worst_frac:.0%} (10 cm tolerance).  '
                    f'Check whether these are the same surface.')
                check['verify_fail'] = True
            else:
                check['note'] = (
                    f'No significant overlap with existing objects '
                    f'(worst {worst_frac:.0%}).')
            checks.append(check)

        elif kind == 'merge':
            source = action.get('source_object')
            # After merge the target holds all observations from both objects.
            # Check if the merged count exceeds HOLD_LIMIT.
            from src.harness.state import HOLD_LIMIT
            merged_held = [k for k, a in proposed_state['assignments'].items()
                           if a.get('object_id') == oid
                           and a.get('status') in ('held', 'assigned')]
            check = {
                'action_index': idx, 'action': 'merge',
                'object_id': oid, 'source_object': source,
                'images': [], 'verify_fail': False,
            }
            if len(merged_held) > HOLD_LIMIT:
                check['note'] = (
                    f'Merge leaves {oid} with {len(merged_held)} held observations, '
                    f'exceeding the limit of {HOLD_LIMIT}.  '
                    f'Re-propose with discard actions to drop '
                    f'{len(merged_held) - HOLD_LIMIT} observation(s).')
                check['verify_fail'] = True
            else:
                check['note'] = (
                    f'Merge accepted; {oid} now holds {len(merged_held)} '
                    f'observation(s), within the limit of {HOLD_LIMIT}.')
            checks.append(check)

    holds = sorted((c for c in checks if c['action'] == 'hold' and c['verify_fail']),
                   key=lambda c: c['hit_rate'])
    for c in holds:
        if len(image_paths) >= MAX_VERIFY_IMAGES:
            c['note'] += (f'  (No image: verify attaches at most {MAX_VERIFY_IMAGES}; '
                          f'use inspect_observations if needed.)')
            continue
        img = _render_backprojection(data, proposed_state, c['object_id'],
                                     c['observation_id'], exclude=[c['observation_id']])
        if img:
            c['images'].append(str(img))
            image_paths.append(img)
    return checks, image_paths


def format_verify_message(checks: list[dict]) -> str:
    """Format checks as a user-turn text block for the model."""
    if not checks:
        return ''
    parts = ['Verify the following before committing:\n']
    for c in checks:
        flag = '⚠' if c.get('verify_fail') else '✓'
        parts.append(
            f'{flag} Action {c["action_index"]} ({c["action"]} → {c["object_id"]}): '
            f'{c["note"]}')
    parts.append(
        '\nIf everything looks correct, call commit_update. '
        'If not, call propose_update with a corrected proposal.')
    return '\n'.join(parts)
