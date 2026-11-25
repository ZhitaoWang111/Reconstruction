"""The outline: exterior walls -> a closed polygon (src.structure.outline), on made-up rooms."""
import numpy as np
import pytest

from src.structure import outline as ol

AX = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0]), np.array([0.0, -1.0])]


def wall(wid, a, b, room_left=True, rms=1.0, jitter=0.0, seed=0):
    """Wall points along a -> b, normal into the room (on the left of a -> b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a)); d = (b - a) / L
    n = np.array([-d[1], d[0]]) if room_left else np.array([d[1], -d[0]])
    t = np.linspace(0.02, 0.98, max(20, int(L * 50)))[:, None]
    pts = a + t * (b - a)
    if jitter:
        pts = pts + np.random.default_rng(seed).normal(0, jitter, (len(pts), 1)) * n
    return {"id": wid, "normal": n.tolist(), "pts": pts, "length_m": L, "rms_cm": rms}


def run(walls, seen=None, centre=None):
    edges = ol.edges_from_walls(walls, AX)
    if seen is not None:
        edges = ol.add_missing_sides(edges, AX, seen)
    pts = np.concatenate([w["pts"] for w in walls])
    return ol.close_polygon(edges, np.asarray(centre if centre is not None else pts.mean(0)))


def corners(res):
    return np.round(np.array(res["corners"]), 2)


def same_ring(got, want):
    """The same polygon, whichever corner it starts at."""
    want = np.asarray(want, float)
    if len(got) != len(want):
        return False
    return any(np.allclose(np.roll(got, -k, axis=0), want, atol=0.03) for k in range(len(got)))


def test_a_rectangle_stays_a_rectangle():
    # each wall seen only in its middle: the corners are where the lines meet
    walls = [wall("S", (1, 0), (3, 0)), wall("E", (4, 1), (4, 2)), wall("N", (3, 3), (1, 3)), wall("W", (0, 2), (0, 1))]
    res = run(walls)
    assert same_ring(corners(res), [(0, 0), (4, 0), (4, 3), (0, 3)])
    assert res["area_m2"] == pytest.approx(12.0, abs=0.05)


def test_a_wall_a_few_degrees_off_the_axis_is_turned_onto_it():
    walls = [wall("S", (1, -0.05), (3, 0.05)),              # ~2.9 deg off: measurement error
             wall("E", (4, 1), (4, 2)), wall("N", (3, 3), (1, 3)), wall("W", (0, 2), (0, 1))]
    res = run(walls)
    s = next(e for e in res["edges"] if e["walls"] == ["S"])
    assert s["direction"] == "snapped" and np.allclose(s["n"], [0, 1])
    assert len(res["corners"]) == 4


def test_an_l_shaped_room_keeps_its_step():
    #  (0,4)-----(2,4)
    #    |          |
    #    |        (2,2)-----(5,2)
    #    |                     |
    #  (0,0)----------------(5,0)
    walls = [wall("S", (0.5, 0), (4.5, 0)), wall("E", (5, 0.5), (5, 1.5)), wall("N1", (4.5, 2), (2.5, 2)),
             wall("E2", (2, 2.5), (2, 3.5)), wall("N2", (1.5, 4), (0.5, 4)), wall("W", (0, 3.5), (0, 0.5))]
    res = run(walls, centre=(1.5, 1.5))
    assert same_ring(corners(res), [(0, 0), (5, 0), (5, 2), (2, 2), (2, 4), (0, 4)])
    assert res["area_m2"] == pytest.approx(14.0, abs=0.1)


def test_a_slanted_exterior_wall_becomes_a_slanted_edge():
    # a 4 x 3 room with its north-east corner cut off by a 45 deg wall from (3,3) to (4,2)
    walls = [wall("S", (0.5, 0), (3.5, 0)), wall("E", (4, 0.5), (4, 1.5)), wall("D", (3.8, 2.2), (3.2, 2.8), rms=1.0),
             wall("N", (2.5, 3), (0.5, 3)), wall("W", (0, 2.5), (0, 0.5))]
    res = run(walls)
    d = next(e for e in res["edges"] if e["walls"] == ["D"])
    assert d["direction"] == "slanted" and d["axis"] is None
    assert same_ring(corners(res), [(0, 0), (4, 0), (4, 2), (3, 3), (0, 3)])


def test_a_short_wall_between_the_thresholds_is_not_trusted_with_a_slant():
    # 15 deg off the axis, but only 0.6 m long: its heading is not reliable enough to keep
    a = np.array([1.0, 0.0]); b = a + 0.6 * np.array([np.cos(np.radians(15)), np.sin(np.radians(15))])
    walls = [wall("S", a, b), wall("E", (4, 1), (4, 2)), wall("N", (3, 3), (1, 3)), wall("W", (0, 2), (0, 1))]
    res = run(walls)
    s = next(e for e in res["edges"] if e["walls"] == ["S"])
    assert s["direction"] == "uncertain" and np.allclose(s["n"], [0, 1])


def test_two_measurements_of_one_wall_are_one_edge():
    walls = [wall("S1", (0.5, 0.00), (1.8, 0.00)), wall("S2", (2.2, 0.08), (3.5, 0.08)),   # 8 cm apart: one wall
             wall("E", (4, 1), (4, 2)), wall("N", (3, 3), (1, 3)), wall("W", (0, 2), (0, 1))]
    res = run(walls)
    assert len(res["corners"]) == 4
    s = next(e for e in res["edges"] if "S1" in e["walls"])
    assert s["walls"] == ["S1", "S2"] or sorted(s["walls"]) == ["S1", "S2"]


def test_a_side_with_no_wall_is_closed_from_seen_space():
    walls = [wall("S", (1, 0), (3, 0)), wall("E", (4, 1), (4, 2)), wall("W", (0, 2), (0, 1))]
    g = np.mgrid[0.05:3.95:0.1, 0.05:2.95:0.1].reshape(2, -1).T          # seen space reaches y = 2.95
    res = run(walls, seen=g)
    n = next(e for e in res["edges"] if e["from"] == "seen_space")
    assert np.allclose(n["n"], [0, -1]) and abs(-n["offset"] - 2.95) < 0.06


def test_the_polygon_is_counter_clockwise_and_edges_run_with_the_room_on_the_left():
    walls = [wall("S", (0.5, 0), (4.5, 0)), wall("E", (5, 0.5), (5, 1.5)), wall("N1", (4.5, 2), (2.5, 2)),
             wall("E2", (2, 2.5), (2, 3.5)), wall("N2", (1.5, 4), (0.5, 4)), wall("W", (0, 3.5), (0, 0.5))]
    res = run(walls, centre=(1.5, 1.5))
    assert res["area_m2"] > 0
    for e in res["edges"]:
        d = e["p1"] - e["p0"]
        left = np.array([-d[1], d[0]])
        assert left @ e["n"] > 0


def test_a_notch_cut_into_one_side():
    # a 6 x 4 room with a 2 x 1 notch cut into the north side between x = 2 and x = 4
    walls = [wall("S", (0.5, 0), (5.5, 0)), wall("E", (6, 0.5), (6, 3.5)),
             wall("N1", (5.5, 4), (4.3, 4)), wall("NE", (4, 3.8), (4, 3.2)), wall("NB", (3.8, 3), (2.2, 3)),
             wall("NW", (2, 3.2), (2, 3.8)), wall("N2", (1.7, 4), (0.5, 4)), wall("W", (0, 3.5), (0, 0.5))]
    res = run(walls, centre=(3, 1.5))
    assert same_ring(corners(res), [(0, 0), (6, 0), (6, 4), (4, 4), (4, 3), (2, 3), (2, 4), (0, 4)])
    assert res["area_m2"] == pytest.approx(22.0, abs=0.1)


def test_a_step_with_no_wall_seen_across_it_gets_a_connector():
    # L room again, but the short wall E2 between the two north walls was never seen
    walls = [wall("S", (0.5, 0), (4.5, 0)), wall("E", (5, 0.5), (5, 1.5)), wall("N1", (4.5, 2), (2.5, 2)),
             wall("N2", (1.5, 4), (0.5, 4)), wall("W", (0, 3.5), (0, 0.5))]
    res = run(walls, centre=(1.5, 1.5))
    conn = [e for e in res["edges"] if e["from"] == "connector"]
    assert len(conn) == 1
    assert res["area_m2"] > 0 and len(res["corners"]) == 6


def test_noisy_walls_still_close_cleanly():
    walls = [wall("S", (1, 0), (3, 0), jitter=0.02, seed=1), wall("E", (4, 1), (4, 2), jitter=0.02, seed=2),
             wall("N", (3, 3), (1, 3), jitter=0.02, seed=3), wall("W", (0, 2), (0, 1), jitter=0.02, seed=4)]
    res = run(walls)
    assert same_ring(corners(res), [(0, 0), (4, 0), (4, 3), (0, 3)])


def _pushed(depth):
    """A 4 x 3 room whose south wall was seen only at its east end; seen space reaches `depth` past it
    along x = 0.5 .. 2.5 (rays ending on a curtain in front of the real wall)."""
    walls = [wall("S", (3.0, 0), (3.8, 0)), wall("E", (4, 1), (4, 2)), wall("N", (3, 3), (1, 3)), wall("W", (0, 2), (0, 1))]
    edges = ol.edges_from_walls(walls, AX)
    poly = ol.close_polygon(edges, np.array([2.0, 1.5]))
    g = np.mgrid[0.05:3.95:0.1, 0.05:2.95:0.1].reshape(2, -1).T
    beyond = np.mgrid[0.525:2.5:0.05, -depth + 0.025:-0.02:0.05].reshape(2, -1).T     # 5 cm cells, as in close.py
    seen = np.r_[g, beyond]
    for e in poly["edges"]:
        L = np.linalg.norm(e["p1"] - e["p0"]); d = (e["p1"] - e["p0"]) / L
        e["observed"] = [(float(np.percentile(np.clip((w["pts"] - e["p0"]) @ d, 0, L), 0.5)),
                          float(np.percentile(np.clip((w["pts"] - e["p0"]) @ d, 0, L), 99.5)))
                         for w in walls if w["id"] in e["walls"]]
    ol.push_out(poly["edges"], seen)
    return poly, ol.pushed_edges(poly["edges"])


def test_seen_space_a_little_past_a_wall_does_not_split_it():
    # 10 cm: within the merge threshold - still one wall, the reach recorded as a note
    poly, extra = _pushed(0.10)
    assert extra == []
    s = next(e for e in poly["edges"] if e["walls"] == ["S"])
    assert s["seen_beyond"] and 0.05 < s["seen_beyond"][0]["depth_m"] < ol.MERGE_STEP


def test_seen_space_well_past_a_wall_splits_off_a_pushed_edge():
    # 40 cm: more than one wall's drift - the unobserved stretch becomes its own edge, further out
    poly, extra = _pushed(0.40)
    assert len(extra) == 1 and extra[0]["from"] == "seen_space"
    assert extra[0]["pushed_by"] > ol.MERGE_STEP
