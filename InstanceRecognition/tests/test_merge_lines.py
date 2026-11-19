"""walls.merge_lines: a run of small depth steps no longer walks across the room."""
import numpy as np
import pytest

from src.structure import walls as wl


def _local(depth, a0, a1):
    # all facing +y (rep 0): depth is the y of the plane, extent along x
    return {"c": np.array([0.0, depth]), "n": np.array([0.0, 1.0]), "a": [-a1, -a0], "rep": 0}


@pytest.fixture(autouse=True)
def _rule(monkeypatch):
    monkeypatch.setattr(wl, "MERGE", "median")


def test_a_chain_of_small_steps_is_cut_where_it_leaves_the_median():
    locs = [_local(d, 0, 1) for d in (0.0, 0.05, 0.10, 0.25, 0.40, 0.55)]   # overlapping, 15 cm steps
    lines = wl.merge_lines(locs, [np.pi / 2])
    assert len(lines) >= 2 and max(abs(r["c"][1] - lines[0][0]["c"][1]) for r in lines[0]) <= wl.DRIFT


def test_far_along_the_wall_needs_a_tighter_depth(monkeypatch):
    near = [_local(0.0, 0, 1), _local(0.2, 0.5, 1.5)]          # overlapping: 20 cm still joins
    far = [_local(0.0, 0, 1), _local(0.2, 3.0, 4.0)]           # 2 m apart: 20 cm > DRIFT_FAR, separate
    assert len(wl.merge_lines(near, [np.pi / 2])) == 1
    assert len(wl.merge_lines(far, [np.pi / 2])) == 2


def test_the_old_rule_still_walks(monkeypatch):
    monkeypatch.setattr(wl, "MERGE", "chain")
    locs = [_local(d, 0, 1) for d in (0.0, 0.25, 0.50, 0.75)]
    assert len(wl.merge_lines(locs, [np.pi / 2])) == 1
