"""Contracts for the SAM3.1 stage's model-free parts.

Everything here runs without torch, opencv or the SAM3 package, which is the
property the split between this module and `load_predictor` exists to give: the
logic that decides *which* masks survive and how they are numbered is ordinary
array handling and can be tested directly.
"""
import numpy as np
import pytest

from src.inference.sam31_segment import (
    PALETTE,
    normalized_xywh_to_xyxy,
    safe_name,
    select_within_budget,
)


def candidate(raw_id, score, **extra):
    return {"raw_candidate_id": raw_id, "sam_score": score, "class_name": "table", **extra}


def test_safe_name_reduces_a_category_to_a_filename():
    assert safe_name("pair of shoes") == "pair_of_shoes"
    assert safe_name("  desk/chair  ") == "desk_chair"
    # A name with nothing usable left must not become an empty filename.
    assert safe_name("///") == "object"


def test_normalized_boxes_become_pixel_corners():
    # SAM reports [x1, y1, w, h] as fractions of the image and the code turns that
    # into absolute corners. Verified against a real prediction, where the recorded
    # box equals the mask's own pixel extent exactly.
    boxes = normalized_xywh_to_xyxy(np.array([[0.25, 0.25, 0.5, 0.5]]), 200, 100)
    assert boxes.tolist() == [[50.0, 25.0, 150.0, 75.0]]


def test_columns_scale_by_width_and_rows_by_height():
    """Swapping the two factors is the easy mistake and shifts every box."""
    boxes = normalized_xywh_to_xyxy(np.array([[0.0, 0.0, 1.0, 1.0]]), 1296, 968)
    assert boxes.tolist() == [[0.0, 0.0, 1296.0, 968.0]]


def test_empty_box_array_stays_empty():
    assert normalized_xywh_to_xyxy(np.zeros((0, 4), dtype=np.float32), 10, 10).shape == (0, 4)


def test_budget_keeps_the_best_scores():
    kept = select_within_budget([candidate(1, 0.10), candidate(2, 0.90), candidate(3, 0.50)], 2)
    assert [item["raw_candidate_id"] for item in kept] == [2, 3]


def test_numbering_follows_prompt_order_not_score_order():
    """Instance ids address masks in the page and the npz, so they must not move."""
    kept = select_within_budget([candidate(1, 0.10), candidate(2, 0.90)], 2)
    assert [item["raw_candidate_id"] for item in kept] == [1, 2]
    assert [item["instance_id"] for item in kept] == [1, 2]


def test_ties_break_deterministically():
    """Equal scores must not let dict ordering decide which mask survives."""
    kept = select_within_budget([candidate(1, 0.5), candidate(2, 0.5)], 1)
    assert [item["raw_candidate_id"] for item in kept] == [1]


def test_budget_does_not_mutate_its_input():
    originals = [candidate(1, 0.1), candidate(2, 0.9)]
    select_within_budget(originals, 1)
    assert "instance_id" not in originals[0] and "instance_id" not in originals[1]


def test_budget_wider_than_candidates_keeps_all():
    kept = select_within_budget([candidate(1, 0.1)], 16)
    assert len(kept) == 1


def test_palette_covers_the_display_budget():
    """Colours cycle by index, so a short palette would repeat within one frame."""
    from src.inference.sam31_segment import DEFAULT_MAX_MASKS

    assert len(PALETTE) >= DEFAULT_MAX_MASKS


def test_heavy_stack_is_not_imported_with_this_module():
    """The harness environment has no torch; importing this module must not need it."""
    import sys

    assert "torch" not in sys.modules
    assert "cv2" not in sys.modules
