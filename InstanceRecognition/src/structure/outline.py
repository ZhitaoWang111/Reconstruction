"""The room's outer boundary from its exterior walls: a closed polygon, not a forced rectangle.

Pure geometry on 2D wall records - no files, no rays - so it can be tested on made-up rooms.
`close.py` feeds it the exterior walls and the seen-space cells and uses what comes back.

What the outline is made of
---------------------------
Each exterior wall becomes an *edge*: a direction and a position. An edge's direction is

* the room's main axis (one of the four `axes`) when the wall faces within SNAP_DEG of it -
  measured wall headings are 2-8 deg off the truth on this scene, and turning them onto the axis
  made them closer to the truth, not further;
* the wall's own measured direction when it faces further than SLANT_DEG off every axis - a real
  slanted wall (W5 here is 34 deg off; no measurement error is anywhere near that);
* in between, the wall's own direction only if the wall is long and flat enough to trust its
  heading (MIN_SLANT_LEN, MAX_SLANT_RMS); otherwise the axis, marked `direction: uncertain`.

An edge's position is the median depth of its wall's points along the edge normal - the same
estimator the old rectangle used - so an edge on the axis lands where the rectangle side did.

Two walls facing the same way within MERGE_STEP of each other are one edge (one surface measured
twice drifts by that much); further apart they are two edges with a step between them.

How the edges become a polygon
------------------------------
The edges are ordered round the room by the direction they face (each points into the room), and
each consecutive pair is joined at the intersection of their lines, however far that is from what
was seen of either wall: an unseen stretch continues its wall. Only where two consecutive edges are
parallel (a step) is a connecting edge inserted, at right angles, and marked inferred. The result is one simple polygon, counter-clockwise.

Where no exterior wall faces some way at all, the side is closed from seen space: an axis edge at
the edge of the space the cameras saw, as the rectangle did (`from: seen_space`).

The outline must hold all the seen space: an edge that seen space reaches past is pushed out to the
seen-space boundary over that stretch (never pulled in - rays stopping short of a wall end on
cabinets and curtains, which say nothing about where the wall is).
"""
from __future__ import annotations

import numpy as np

SNAP_DEG = 10.0        # within this of an axis: the wall is turned onto it
SLANT_DEG = 20.0       # beyond this off every axis: the wall keeps its own direction
MIN_SLANT_LEN = 1.0    # m; between the two, a wall this long ...
MAX_SLANT_RMS = 3.0    # cm; ... and this flat keeps its own direction
MERGE_STEP = 0.15      # m; same-facing edges closer than this are one edge


def heading(n) -> float:
    return float(np.arctan2(n[1], n[0]))


def wrap(a: float, period: float = 2 * np.pi) -> float:
    return float((a + period / 2) % period - period / 2)


def classify(n, axes, length_m: float, rms_cm: float) -> dict:
    """How a wall facing `n` is placed: on axis k, or slanted with its own direction."""
    devs = [abs(np.degrees(wrap(heading(n) - heading(e)))) for e in axes]
    k = int(np.argmin(devs))
    dev = devs[k]
    if dev <= SNAP_DEG:
        return {"axis": k, "dev_deg": dev, "direction": "snapped"}
    if dev >= SLANT_DEG:
        return {"axis": None, "dev_deg": dev, "direction": "slanted"}
    if length_m >= MIN_SLANT_LEN and rms_cm <= MAX_SLANT_RMS:
        return {"axis": None, "dev_deg": dev, "direction": "slanted"}
    return {"axis": k, "dev_deg": dev, "direction": "uncertain"}


def edges_from_walls(walls: list, axes: list) -> list:
    """One edge per exterior wall (merged where two walls are one surface).

    `walls` entries need: id, normal (unit, into the room), pts (N, 2), length_m, rms_cm.
    """
    edges = []
    for w in walls:
        n0 = np.asarray(w["normal"], dtype=np.float64)
        c = classify(n0, axes, w["length_m"], w["rms_cm"])
        n = axes[c["axis"]] if c["axis"] is not None else n0 / np.linalg.norm(n0)
        u = np.array([-n[1], n[0]])
        pts = np.asarray(w["pts"], dtype=np.float64)
        al = pts @ u
        edges.append({"n": n, "offset": float(np.median(pts @ n)), "walls": [w["id"]], "axis": c["axis"],
                      "direction": c["direction"], "dev_deg": c["dev_deg"],
                      "span": [float(np.percentile(al, 0.5)), float(np.percentile(al, 99.5))],
                      "pts": pts, "from": "observed"})
    # One surface measured twice: same facing, depths within MERGE_STEP - unless another exterior
    # wall stands in the gap between them along the wall. Two collinear stretches with a notch
    # between them are two walls (the notch's walls are in the gap); a wall seen in two pieces with
    # nothing between them is one.
    def blocked(m, e):
        u = np.array([-m["n"][1], m["n"][0]])
        lo = min(m["span"][1], e["span"][1]); hi = max(m["span"][0], e["span"][0])
        if hi <= lo:                       # they overlap along the wall: nothing can be between
            return False
        for o in edges:
            if o is e or any(w in m["walls"] for w in o["walls"]):
                continue
            t = o["pts"] @ u
            d = o["pts"] @ m["n"] - m["offset"]
            if ((t > lo + 0.05) & (t < hi - 0.05) & (np.abs(d) < 1.5)).mean() > 0.5:
                return True
        return False

    merged = []
    for e in sorted(edges, key=lambda e: -len(e["pts"])):
        for m in merged:
            if (np.dot(m["n"], e["n"]) > np.cos(np.radians(2.0)) and abs(m["offset"] - e["offset"]) < MERGE_STEP
                    and not blocked(m, e)):
                pts = np.r_[m["pts"], e["pts"]]
                u = np.array([-m["n"][1], m["n"][0]])
                m.update(pts=pts, offset=float(np.median(pts @ m["n"])), walls=m["walls"] + e["walls"],
                         span=[min(m["span"][0], e["span"][0]), max(m["span"][1], e["span"][1])])
                break
        else:
            merged.append(dict(e))
    return merged


def add_missing_sides(edges: list, axes: list, seen_pts: np.ndarray) -> list:
    """An axis no edge faces at all is closed at the edge of the seen space (0.5 percentile)."""
    out = list(edges)
    for k, e in enumerate(axes):
        if any(np.dot(x["n"], e) > np.cos(np.radians(SLANT_DEG)) for x in edges):
            continue
        u = np.array([-e[1], e[0]])
        al = seen_pts @ u
        out.append({"n": e, "offset": float(np.percentile(seen_pts @ e, 0.5)), "walls": [], "axis": k,
                    "direction": "snapped", "dev_deg": 0.0, "span": [float(al.min()), float(al.max())],
                    "pts": np.zeros((0, 2)), "from": "seen_space"})
    return out


def _intersect(a: dict, b: dict):
    """Point on both edge lines (x . n = offset), or None if they are within 5 deg of parallel."""
    A = np.array([a["n"], b["n"]])
    if abs(np.linalg.det(A)) < np.sin(np.radians(5.0)):
        return None
    return np.linalg.solve(A, np.array([a["offset"], b["offset"]]))


def _dir(e: dict) -> np.ndarray:
    """The direction an edge runs with the room on its left: the inward normal turned clockwise."""
    return np.array([e["n"][1], -e["n"][0]])


def _along(e: dict, p) -> float:
    return float(np.asarray(p) @ _dir(e))


def _extent(e: dict) -> tuple:
    """Where along its running direction the edge's wall was observed: (start, end)."""
    if len(e["pts"]):
        t = e["pts"] @ _dir(e)
        return float(np.percentile(t, 0.5)), float(np.percentile(t, 99.5))
    return e["span_run"]


def _ends(e: dict):
    """The observed start and end points of an edge, running with the room on its left."""
    t0, t1 = _extent(e)
    base = e["offset"] * e["n"]
    return base + t0 * _dir(e), base + t1 * _dir(e)


def order_edges(edges: list, centre) -> list:
    """The edges in walking order round the room (counter-clockwise), by chaining ends to starts.

    Ordering by the angle of each edge's middle seen from a centre only works for a convex room: in
    a notch the walls crowd together in angle and come out shuffled. Walking does not need a centre.
    From each edge, the next one is the edge whose observed start is nearest to this edge's
    observed end among those that turn left or run on (cross >= 0 within tolerance) - a walk round
    the inside of a room never turns more than 180 deg at a corner. The walk starts at the longest
    edge and each edge is used once.
    """
    if len(edges) <= 2:
        return list(edges)
    left = list(range(len(edges)))
    i0 = max(left, key=lambda i: _extent(edges[i])[1] - _extent(edges[i])[0])
    ring = [edges[i0]]; left.remove(i0)
    while left:
        a = ring[-1]
        _, end_a = _ends(a)
        da = _dir(a)

        def cost(b):
            start_b, _ = _ends(b)
            gap = float(np.linalg.norm(start_b - end_a))
            # going backwards along a (b starts well behind a's end) is not the next wall
            back = float((start_b - end_a) @ da)
            turn = da[0] * _dir(b)[1] - da[1] * _dir(b)[0]          # + = left turn
            right_turn = turn < -0.2 and np.dot(da, _dir(b)) < 0.95  # an inside corner: allowed,
            return gap + (2.0 if back < -1.0 else 0.0) + (0.25 if right_turn else 0.0)
        j = min(left, key=lambda i: cost(edges[i]))
        ring.append(edges[j]); left.remove(j)
    # walking direction: counter-clockwise has positive area
    pts = np.array([_ends(e)[0] for e in ring])
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    return ring if area >= 0 else [ring[0]] + ring[:0:-1]


def close_polygon(edges: list, centre) -> dict:
    """Join consecutive edges into one polygon; insert connecting edges where they do not meet.

    Each edge runs with the room on its left. Consecutive edges a, b meet at the intersection of
    their lines. Two parallel edges (a step) cannot, and a connecting edge at right angles is
    inserted from a's end.

    Returns {"corners": [...], "edges": [...] each with p0 -> p1, "area_m2"}, counter-clockwise.
    """
    centre = np.asarray(centre, dtype=np.float64)
    for e in edges:
        e.setdefault("span_run", None)
        if e["span_run"] is None:
            e["span_run"] = (float(-e["span"][1]), float(-e["span"][0]))    # span was measured along +90 deg
    ring = order_edges(edges, centre)
    full = []
    for i, a in enumerate(ring):
        b = ring[(i + 1) % len(ring)]
        full.append(a)
        p = _intersect(a, b)
        # Two edges that are not parallel meet where their lines cross, however far that is from
        # what was seen of either wall: an unseen stretch continues the wall it belongs to (the
        # stretch is marked inferred downstream). Only two parallel edges - a step - cannot meet,
        # and only then is a connecting edge inserted. A corner that would fall *behind* the start
        # of a (the walk would turn back on itself) is a step in disguise and is treated as one.
        ok = p is not None
        if ok:
            a0, a1 = _extent(a); b0, b1 = _extent(b)
            ok = _along(a, p) >= a0 - 0.30 and _along(b, p) <= b1 + 0.30
        if not ok:
            # A connecting edge from a's end to b's start: a walk a -> connector -> b. Its running
            # direction is from where a stops to where b begins, so its inward normal (room on the
            # left) is that direction turned counter-clockwise - no guessing which way the step goes.
            # It is drawn at right angles to a (a step), through a's end.
            _, a1 = _extent(a); b0, _ = _extent(b)
            end_a = a["offset"] * a["n"] + a1 * _dir(a)
            start_b = b["offset"] * b["n"] + b0 * _dir(b)
            run = (start_b - end_a) @ a["n"]                 # how far b lies off a, across a
            d = -a["n"] if run < 0 else a["n"]                # the connector runs across a, towards b
            n = np.array([-d[1], d[0]])                       # room on its left
            t = float(end_a @ _dir({"n": n}))
            full.append({"n": n, "offset": float(end_a @ n), "walls": [], "axis": a["axis"], "direction": "connector",
                         "dev_deg": 0.0, "span": [0.0, 0.0], "span_run": (t, t), "pts": np.zeros((0, 2)),
                         "from": "connector"})
    corners = []
    for i, a in enumerate(full):
        b = full[(i + 1) % len(full)]
        p = _intersect(a, b)
        if p is None:
            p = a["offset"] * a["n"] + _extent(a)[1] * _dir(a)
        corners.append(np.asarray(p))
    out_edges = []
    for i, e in enumerate(full):
        out_edges.append({**{k: v for k, v in e.items()}, "p0": corners[i - 1], "p1": corners[i],
                          "n": np.asarray(e["n"])})
    C = np.array([e["p0"] for e in out_edges])
    area = 0.5 * float(np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1]))
    return {"corners": [e["p0"] for e in out_edges], "edges": out_edges, "area_m2": area}


def push_out(edges: list, seen_pts: np.ndarray, step: float = 0.20, margin: float = 0.02) -> None:
    """Seen space the outline must hold: where it reaches past an edge, move that stretch out.

    Only outward, and only on stretches no wall was seen on: an observed stretch stays where its
    wall was measured. Recorded per edge as `pushed`: [(a0, a1, depth_m), ...] along the edge.
    """
    for e in edges:
        e["pushed"] = []
        if e["from"] == "connector":
            continue
        L = float(np.linalg.norm(e["p1"] - e["p0"]))
        if L < step:
            continue
        d = (e["p1"] - e["p0"]) / L
        a = (seen_pts - e["p0"]) @ d
        q = (seen_pts - e["p0"]) @ e["n"]          # + = inside the room
        for s in np.arange(0.0, L, step):
            if any(o0 - 0.05 <= s and s + step <= o1 + 0.05 for o0, o1 in e.get("observed", [])):
                continue
            m = (a >= s) & (a < s + step) & (q < -margin) & (q > -0.6)
            if m.sum() >= 3:
                e["pushed"].append((float(s), float(min(L, s + step)), float(-np.percentile(q[m], 5))))


def inside(poly, pts: np.ndarray, grow: float = 0.0) -> np.ndarray:
    """Points inside the polygon (even-odd rule); with grow > 0, also points within grow of its edges."""
    P = np.asarray(poly, dtype=np.float64); X = np.asarray(pts, dtype=np.float64)
    res = np.zeros(len(X), bool)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        cond = (a[1] > X[:, 1]) != (b[1] > X[:, 1])
        xs = a[0] + (X[:, 1] - a[1]) * (b[0] - a[0]) / np.where(b[1] == a[1], 1e-12, b[1] - a[1])
        res ^= cond & (X[:, 0] < xs)
    if grow > 0:
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            e = b - a; t = np.clip(((X - a) @ e) / max(e @ e, 1e-12), 0, 1)
            res |= np.linalg.norm(X - (a + t[:, None] * e), axis=1) <= grow
    return res


# Seen space reaching past an edge splits it off as a new, pushed-out edge only if it reaches further
# than MERGE_STEP: the same threshold that makes two same-facing stretches one wall. Less than that
# is within one surface's depth drift (up to ~24 cm between frames here) - two edges that close
# would be merged straight back - so the edge stays one wall and the reach is recorded as a note.
MIN_PUSH = MERGE_STEP
MIN_PUSH_LEN = 0.60    # m; ... over at least this long a stretch (a few cells past a wall is noise)


def pushed_edges(edges: list) -> list:
    """New edges for the stretches `push_out` found on each edge.

    All pushed stretches of one edge are one run at one depth (the median), from the first to the
    last: seen space reaching past an unobserved stretch is usually rays ending on a curtain or the
    like in front of the wall, and tracing it cell by cell would only draw that curtain as a saw
    tooth. A run becomes an edge only if it is MIN_PUSH_LEN long and deeper than MIN_PUSH (= the
    merge threshold); a shallower one is recorded on the edge as `seen_beyond`. Its edge is parallel
    to the edge it came from, `from: seen_space`, with no wall behind it - all of it is inferred.
    """
    out = []
    for e in edges:
        p = e.get("pushed", [])
        if not p:
            continue
        a0 = min(x[0] for x in p); a1 = max(x[1] for x in p)
        # stop the run at an observed stretch: an observed wall is never moved
        for o0, o1 in e.get("observed", []):
            if o0 < a1 and o1 > a0:
                if o0 > a0:
                    a1 = min(a1, o0)
                else:
                    a0 = max(a0, o1)
        runs = [[a0, a1, [x[2] for x in p if a0 <= x[0] < a1]]] if a1 > a0 else []
        L = float(np.linalg.norm(np.asarray(e["p1"]) - np.asarray(e["p0"])))
        d = (np.asarray(e["p1"]) - np.asarray(e["p0"])) / max(L, 1e-9)
        for a0, a1, depths in runs:
            depth = float(np.median(depths)) if depths else 0.0
            if a1 - a0 < MIN_PUSH_LEN:
                continue
            if depth < MIN_PUSH:
                # within one wall's depth drift: not a separate wall, but say that seen space
                # reached past this stretch, and by how much
                e.setdefault("seen_beyond", []).append({"from_m": round(a0, 3), "to_m": round(a1, 3),
                                                        "depth_m": round(depth, 3)})
                continue
            n = np.asarray(e["n"])
            q0 = np.asarray(e["p0"]) + a0 * d - depth * n
            q1 = np.asarray(e["p0"]) + a1 * d - depth * n
            run_t = (float(q0 @ _dir(e)), float(q1 @ _dir(e)))
            out.append({"n": n, "offset": float(q0 @ n), "walls": [], "axis": e["axis"], "direction": e["direction"],
                        "dev_deg": e["dev_deg"], "span": [0.0, 0.0], "span_run": run_t, "pts": np.zeros((0, 2)),
                        "from": "seen_space", "pushed_by": round(depth, 3)})
    return out
