from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.harness.history import choose_representative, historical_bbox_view


def make_data(tmp_path, frames, observations):
    paths = []
    for seq, frame in enumerate(frames):
        path = tmp_path / f'{seq}.png'
        Image.fromarray(frame.astype(np.uint8)).convert('RGB').save(path)
        paths.append(path)
    return SimpleNamespace(items=observations, out=tmp_path / 'output', rgb_view=lambda seq: paths[seq])


def striped(size=100, amplitude=255):
    return (np.indices((size, size))[1] % 2 * amplitude).astype(np.uint8)


def test_select_clear_large_box_and_frame_cache(tmp_path):
    data = make_data(tmp_path, [striped(), np.zeros((100, 100))], {
        'a': {'seq': 0, 'bbox_xyxy': [0, 0, 10, 10]},
        'b': {'seq': 0, 'bbox_xyxy': [0, 0, 50, 50]},
        'c': {'seq': 1, 'bbox_xyxy': [0, 0, 100, 100]},
    })
    oid, metrics = choose_representative(data, ['c', 'a', 'b'])
    assert oid == 'b'
    assert metrics['normalized_bbox_area'] == .25
    assert metrics['clarity'] > 0
    assert len(data._history_frame_metrics) == 2
    data.rgb_view = lambda seq: pytest.fail('clarity must be cached per frame')
    assert choose_representative(data, ['a', 'b'])[0] == 'b'


def test_normalized_area_and_stable_id_tie(tmp_path):
    data = make_data(tmp_path, [np.zeros((100, 100)), np.zeros((200, 200))], {
        'b': {'seq': 0, 'bbox_xyxy': [0, 0, 50, 50]},
        'a': {'seq': 1, 'bbox_xyxy': [0, 0, 100, 100]},
        'c': {'seq': 1, 'bbox_xyxy': [0, 0, 80, 80]},
    })
    assert choose_representative(data, ['c', 'b', 'a'])[0] == 'a'
    assert choose_representative(data, ['a', 'b', 'c'])[0] == 'a'


def test_no_bbox_fallback_and_empty_candidates(tmp_path):
    data = make_data(tmp_path, [np.zeros((100, 100)), striped()], {
        'a': {'seq': 0, 'bbox_xyxy': None},
        'b': {'seq': 1, 'bbox_xyxy': [5, 5, 5, 10]},
    })
    oid, metrics = choose_representative(data, ['a', 'b'])
    assert oid == 'b' and metrics['missing_bbox']
    assert metrics['normalized_bbox_area'] is None
    with pytest.raises(ValueError, match='No historical'):
        choose_representative(data, [])
    path = historical_bbox_view(data, 'Object2', oid)
    with Image.open(path) as image:
        assert image.size == (100, 100)
        np.testing.assert_allclose(np.asarray(image)[50:, :, 0], striped()[50:], atol=3)


def test_view_full_rgb_single_original_box_without_tint(tmp_path):
    data = make_data(tmp_path, [np.full((120, 160, 3), [40, 80, 120])], {
        'a': {'seq': 0, 'bbox_xyxy': [10, 10, 80, 100]},
    })
    path = historical_bbox_view(data, 'Object1', 'a')
    with Image.open(path) as image:
        assert image.size == (160, 120)
        pixels = np.asarray(image)
        np.testing.assert_allclose(pixels[60, 45], [40, 80, 120], atol=3)
        np.testing.assert_allclose(pixels[60, 120], [40, 80, 120], atol=3)
        assert pixels[60, 10, 0] > 220
    assert data.items['a']['bbox_xyxy'] == [10, 10, 80, 100]
