"""On-demand historical representatives: one full RGB with one original-mask bbox.

Clarity is the variance of scipy's discrete Laplacian on full-frame grayscale
RGB (0..255). Within candidates having valid boxes, scores >= half the best
score form the clear pool; maximize normalized bbox area, then clarity, then
lexicographically smallest observation ID. This relative heuristic does not
assert absolute image quality. If no valid box exists, retain the clearest
full RGB and report missing_bbox in selection metadata.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import laplace


CLEAR_RELATIVE_THRESHOLD = 0.5


def _frame_metrics(data, seq):
    cache = getattr(data, '_history_frame_metrics', None)
    if cache is None:
        cache = data._history_frame_metrics = {}
    if seq not in cache:
        with Image.open(data.rgb_view(seq)) as source:
            rgb = source.convert('RGB')
            gray = np.asarray(rgb.convert('L'), dtype=np.float64)
            cache[seq] = (float(laplace(gray, mode='reflect').var()), rgb.size)
    return cache[seq]


def _valid_bbox(observation, size):
    box = observation.get('bbox_xyxy')
    if box is None or len(box) != 4:
        return None
    values = np.asarray(box, dtype=np.float64)
    width, height = size
    if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
            or not (0 <= values[0] < values[2] <= width)
            or not (0 <= values[1] < values[3] <= height)):
        return None
    return tuple(int(v) for v in values)


def choose_representative(data, ids):
    """Return (observation_id, JSON-safe selection metrics); caller bounds ids."""
    candidates = []
    for oid in sorted(set(ids)):
        observation = data.items[oid]
        clarity, size = _frame_metrics(data, observation['seq'])
        box = _valid_bbox(observation, size)
        area = ((box[2] - box[0]) * (box[3] - box[1]) / (size[0] * size[1])) if box else None
        candidates.append({'observation_id': oid, 'clarity': clarity,
                           'normalized_bbox_area': area, 'missing_bbox': box is None})
    if not candidates:
        raise ValueError('No historical observations to select')
    valid = [c for c in candidates if not c['missing_bbox']]
    best_clarity = max(c['clarity'] for c in (valid or candidates))
    if valid:
        pool = [c for c in valid if c['clarity'] >= CLEAR_RELATIVE_THRESHOLD * best_clarity]
        selected = min(pool, key=lambda c: (-c['normalized_bbox_area'], -c['clarity'], c['observation_id']))
        rule = 'relative_clear_pool_then_normalized_bbox_area'
    else:
        selected = min(candidates, key=lambda c: (-c['clarity'], c['observation_id']))
        rule = 'no_valid_bbox_clearest_full_rgb'
    metrics = {**selected, 'selection_rule': rule, 'clarity_method': 'full_rgb_grayscale_laplacian_variance',
               'clear_relative_threshold': CLEAR_RELATIVE_THRESHOLD, 'best_candidate_clarity': best_clarity,
               'candidate_count': len(candidates), 'valid_bbox_candidate_count': len(valid)}
    return selected['observation_id'], metrics


def historical_bbox_view(data, object_id, observation_id):
    """Render full-frame RGB with only this original-mask bbox and Object ID.

    The original observation/mask is never modified or fused. Missing boxes
    leave RGB intact apart from the Object ID at the top-left; selection
    metadata conveys missing_bbox. A JPEG is returned for the existing client.
    """
    observation = data.items[observation_id]
    with Image.open(data.rgb_view(observation['seq'])) as source:
        image = source.convert('RGB')
    box = _valid_bbox(observation, image.size)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', max(17, image.width // 45))
    except OSError:
        font = ImageFont.load_default()
    x, y = box[:2] if box else (0, 0)
    if box:
        # Stored maxima are exclusive; PIL rectangle maxima are inclusive.
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=(255, 210, 0), width=3)
    label = str(object_id)
    bounds = draw.textbbox((0, 0), label, font=font)
    x = min(x, max(0, image.width - (bounds[2] - bounds[0])))
    y = min(y, max(0, image.height - (bounds[3] - bounds[1])))
    position = (x - bounds[0], y - bounds[1])
    draw.rectangle(draw.textbbox(position, label, font=font), fill=(0, 0, 0))
    draw.text(position, label, font=font, fill=(255, 255, 255))
    key = hashlib.sha256(f'{object_id}\0{observation_id}'.encode()).hexdigest()[:20]
    path = Path(data.out) / 'evidence' / f'history_bbox_{key}.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95, subsampling=0)
    return path
