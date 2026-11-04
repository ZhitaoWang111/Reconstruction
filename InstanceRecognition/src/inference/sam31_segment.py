"""SAM3.1 adapter: one RGB frame and GPT text prompts -> independent binary masks."""
import contextlib
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src import config

DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_MASKS = 16
PALETTE = [
    (255, 99, 71), (30, 144, 255), (50, 205, 50), (255, 193, 7),
    (186, 85, 211), (0, 206, 209), (255, 105, 180), (255, 140, 0),
    (70, 130, 180), (154, 205, 50), (220, 20, 60), (64, 224, 208),
    (138, 43, 226), (210, 180, 140), (0, 191, 255), (240, 128, 128),
]


def safe_name(label):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', label.strip()).strip('_') or 'object'


def normalized_xywh_to_xyxy(boxes, width, height):
    result = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    result *= [width, height, width, height]
    result[:, 2:] += result[:, :2]
    return result


def select_within_budget(candidates, max_masks):
    selected = sorted(candidates, key=lambda x: (-x['sam_score'], x['raw_candidate_id']))[:max_masks]
    return [dict(item, instance_id=i + 1)
            for i, item in enumerate(sorted(selected, key=lambda x: x['raw_candidate_id']))]


def load_predictor(checkpoint, threshold=DEFAULT_THRESHOLD):
    checkpoint = Path(checkpoint).resolve(strict=True)
    # Import only on real inference; the harness and offline tests need no torch.
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            f'SAM runtime missing in {sys.executable}: {exc}. '
            f'Use {config.SAM31_PYTHON}; see requirements-runtime.txt.'
        ) from exc
    config.TORCH_KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('PYTORCH_KERNEL_CACHE_PATH', str(config.TORCH_KERNEL_CACHE))
    return build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint), max_num_objects=DEFAULT_MAX_MASKS,
        multiplex_count=DEFAULT_MAX_MASKS, use_fa3=False, use_rope_real=False,
        compile=False, warm_up=False, async_loading_frames=False,
        default_output_prob_thresh=threshold,
    )


@contextlib.contextmanager
def sam3_session(predictor, source):
    # The public API expects a directory of frames. Copy JPEG bytes without re-encoding.
    import shutil
    with tempfile.TemporaryDirectory(prefix='sam31-frame-') as folder:
        shutil.copyfile(source, Path(folder) / '000000.jpg')
        session = predictor.handle_request({'type': 'start_session', 'resource_path': folder})['session_id']
        try:
            yield session
        finally:
            predictor.handle_request({'type': 'close_session', 'session_id': session})


def segment(predictor, source, categories, output, threshold=DEFAULT_THRESHOLD,
            max_masks=DEFAULT_MAX_MASKS):
    if not 0 <= threshold <= 1 or not 1 <= max_masks <= 32:
        raise ValueError('Invalid mask settings')
    source, output = Path(source).resolve(), Path(output)
    with Image.open(source) as image:
        if image.format != 'JPEG':
            raise ValueError('Expected an original JPEG RGB frame')
        width, height = image.size
    (output / 'raw_masks').mkdir(parents=True, exist_ok=True)
    candidates, prompts = [], []
    started = time.monotonic()
    with sam3_session(predictor, source) as session:
        for index, category in enumerate(categories):
            tick = time.monotonic()
            # SAM3.1's text add_prompt resets the previous prompt state internally.
            response = predictor.handle_request({
                'type': 'add_prompt', 'session_id': session, 'frame_index': 0,
                'text': category['sam_prompt'], 'output_prob_thresh': threshold,
            })['outputs']
            masks = np.asarray(response['out_binary_masks'], dtype=bool)
            scores = np.asarray(response['out_probs']).reshape(-1)
            boxes = normalized_xywh_to_xyxy(response['out_boxes_xywh'], width, height)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.size == 0:
                masks = np.empty((0, height, width), dtype=bool)
            if masks.shape != (len(scores), height, width) or len(boxes) != len(scores):
                raise ValueError('SAM text output shape mismatch')
            if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
                raise ValueError('Nonfinite SAM output')
            ids, empty = [], []
            for candidate_index, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
                if not mask.any():
                    empty.append(candidate_index)
                    continue
                raw_id = len(candidates) + 1
                mask_file = f'raw_masks/{raw_id:03d}_{safe_name(category["category"])}.png'
                Image.fromarray(mask.astype(np.uint8) * 255).save(output / mask_file)
                candidates.append({
                    'raw_candidate_id': raw_id, 'class_name': category['category'],
                    'class_id': index, 'prompt_index': index, 'sam_prompt': category['sam_prompt'],
                    'gpt_certainty': category['certainty'], 'visible_scope': category['visible_scope'],
                    'evidence': category['evidence'], 'ambiguity': category['ambiguity'],
                    'sam_score': float(score), 'sam_box_xyxy': box.round(2).tolist(),
                    'sam_prompt_type': 'text', 'mask_file': mask_file, 'mask_pixels': int(mask.sum()),
                })
                ids.append(raw_id)
            prompts.append({**category, 'prompt_index': index, 'raw_candidate_ids': ids,
                            'empty_candidate_indices': empty, 'elapsed_seconds': time.monotonic() - tick})
    selected = select_within_budget(candidates, max_masks)
    kept = {item['raw_candidate_id'] for item in selected}
    return {
        'stage': 'GPT labels -> SAM3.1 text masks', 'source': str(source),
        'image_shape_hwc': [height, width, 3], 'instances': selected, 'instance_count': len(selected),
        'raw_candidates': candidates, 'raw_candidate_count': len(candidates), 'prompts': prompts,
        'excluded_by_budget': [item['raw_candidate_id'] for item in candidates if item['raw_candidate_id'] not in kept],
        'unique_classes': sorted({item['class_name'] for item in selected}),
        'policy': {'sam_output_prob_threshold': threshold, 'max_selected_masks': max_masks,
                   'selection': 'top scores; ties by candidate order; all raw masks retained',
                   'cross_category_nms': False, 'temporal_propagation': False},
        'elapsed_seconds': time.monotonic() - started,
    }
