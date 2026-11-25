"""Cut every wall into independent segments and describe the prism each one builds.

A wall is cut at the centre line of every wall that butts onto its room-side face, so an
exterior wall becomes one body per room it faces. Mitred corners are never cut: those
footprint corners already carry the mitre, and the segment reuses them verbatim.

The output of this module is plain data (no Blender, no numpy types), so stage 1 can be
checked without starting Blender and the Blender script only has to build what it says.
"""
from __future__ import annotations

import math


def _unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _mul(v, k):
    return (v[0] * k, v[1] * k)


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def wall_axes(wall: dict):
    """start point, unit direction along the wall (start -> end), unit normal into the back."""
    s = (wall["start_floor"][0], wall["start_floor"][1])
    e = (wall["end_floor"][0], wall["end_floor"][1])
    return s, _unit(_sub(e, s)), _unit((wall["normal_to_back"][0], wall["normal_to_back"][1]))


def butt_cuts(doc: dict) -> dict[str, list[float]]:
    """Along-wall positions where another wall butts onto this wall's room-side face.

    The cut is the centre line of the butting wall, taken from its own footprint corners,
    so a thickness change on either wall moves the cut the way the geometry does.
    """
    walls = {w["id"]: w for w in doc["walls"]}
    cuts = {wid: [] for wid in walls}
    for w in doc["walls"]:
        for end in ("start", "end"):
            join = w["joins"].get(end)
            if not join or join["type"] != "butt":
                continue
            s, u, _ = wall_axes(walls[join["wall"]])
            fp = walls[w["id"]]["footprint_floor"]
            pts = (fp[0], fp[3]) if end == "start" else (fp[1], fp[2])
            along = sorted(_dot(_sub((p[0], p[1]), s), u) for p in pts)
            cuts[join["wall"]].append((along[0] + along[1]) / 2)
    return cuts


def spans(wall: dict, cuts: dict[str, list[float]]):
    """(a0, a1, id) for every segment of one wall."""
    edges = [0.0] + sorted(cuts[wall["id"]]) + [wall["length_m"]]
    out = []
    for i in range(len(edges) - 1):
        sid = wall["id"] if len(edges) == 2 else f"{wall['id']}_{i + 1}"
        out.append((edges[i], edges[i + 1], sid))
    return out


def plan(doc: dict, height_m: float | None = None, opening_inset_m: float = 0.0) -> dict:
    """Every segment with the prism it builds, and the volume that prism must have.

    ``height_m`` overrides the handoff's storey height for the whole model (the handoff's
    2.80 m is an assumption); per-segment heights arrive with the appearance stage.
    ``opening_inset_m`` narrows every opening by this much on each side (left and right, not the
    top): a handoff that takes the opening from an annotated door frame gives its outer edge,
    casing included. The handoff's own range is kept on the opening as ``handoff_m``.
    """
    cuts = butt_cuts(doc)
    if opening_inset_m:
        doc = {**doc, "walls": [{**w, "openings": [_inset(o, opening_inset_m, w["id"]) for o in w["openings"]]}
                                for w in doc["walls"]]}
    segments = []
    dropped = []
    for w in doc["walls"]:
        h = float(height_m if height_m is not None else w["height_m"])
        s, u, n = wall_axes(w)
        t = float(w["thickness_m"])
        fp = [tuple(p[:2]) for p in w["footprint_floor"]]
        sp = spans(w, cuts)
        for i, (a0, a1, sid) in enumerate(sp):
            first, last = i == 0, i == len(sp) - 1
            front = (fp[0] if first else _add(s, _mul(u, a0)),
                     fp[1] if last else _add(s, _mul(u, a1)))
            back = (fp[3] if first else _add(_add(s, _mul(u, a0)), _mul(n, t)),
                    fp[2] if last else _add(_add(s, _mul(u, a1)), _mul(n, t)))
            own = [o for o in w["openings"] if a0 < o["from_m"] and o["to_m"] < a1]
            for o in w["openings"]:
                if a0 < o["from_m"] < a1 < o["to_m"] or o["from_m"] < a0 < o["to_m"] < a1:
                    entry = {"wall": w["id"], "from_m": o["from_m"], "to_m": o["to_m"]}
                    if entry not in dropped:            # an opening is seen from both sides of its cut
                        dropped.append(entry)
            area = _quad_area(front[0], front[1], back[1], back[0])
            hole = sum((o["to_m"] - o["from_m"]) * (o["z_floor"][1] - o["z_floor"][0])
                       for o in own)
            segments.append({
                "id": sid, "wall": w["id"], "role": w["role"], "a0": a0, "a1": a1,
                "length_m": a1 - a0, "thickness_m": t, "height_m": h,
                "front": [list(p) for p in front], "back": [list(p) for p in back],
                "openings": [{"from_m": o["from_m"], "to_m": o["to_m"], "z_floor": list(o["z_floor"]),
                              **({"handoff_m": o["handoff_m"]} if "handoff_m" in o else {})} for o in own],
                "footprint_area_m2": area,
                "hole_volume_m3": hole * t,
                "expected_volume_m3": area * h - hole * t,
            })
    return {"schema": "structuregen/plan/v1", "scene_id": doc["scene_id"],
            "frame": doc["coordinate_frame"], "height_m": height_m, "opening_inset_m": opening_inset_m,
            "cuts": {k: sorted(v) for k, v in cuts.items()},
            "dropped_openings": dropped, "segments": segments}


def _inset(o: dict, inset: float, wall: str) -> dict:
    lo, hi = o["from_m"] + inset, o["to_m"] - inset
    if hi - lo < 0.3:
        raise ValueError(f"{wall}: opening {o['from_m']:.3f}-{o['to_m']:.3f} is {hi - lo:.3f} m wide after "
                         f"an inset of {inset} m per side")
    return {**o, "from_m": round(lo, 4), "to_m": round(hi, 4), "handoff_m": [o["from_m"], o["to_m"]]}


def _quad_area(p0, p1, p2, p3) -> float:
    a = 0.0
    quad = (p0, p1, p2, p3)
    for k in range(4):
        x0, y0 = quad[k]
        x1, y1 = quad[(k + 1) % 4]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2


def check(planned: dict) -> list[str]:
    """Problems with the plan itself: segment count, opening placement, positive volume."""
    bad = []
    for d in planned.get("dropped_openings", []):
        bad.append(f"{d['wall']}: opening {d['from_m']:.3f}-{d['to_m']:.3f} straddles a butt cut; "
                   f"it would be left solid")
    for seg in planned["segments"]:
        if seg["expected_volume_m3"] <= 0:
            bad.append(f"{seg['id']}: expected volume {seg['expected_volume_m3']:.4f} <= 0")
        f0, f1 = seg["front"]
        u = _unit(_sub(f1, f0))
        back_lo = seg["a0"] + _dot(_sub(seg["back"][0], f0), u)       # where the back face starts and ends,
        back_hi = seg["a0"] + _dot(_sub(seg["back"][1], f0), u)       # in the wall's along coordinate
        for o in seg["openings"]:
            if not (seg["a0"] < o["from_m"] and o["to_m"] < seg["a1"]):
                bad.append(f"{seg['id']}: opening {o['from_m']:.3f}-{o['to_m']:.3f} straddles the cut")
            if not (back_lo < o["from_m"] and o["to_m"] < back_hi):
                bad.append(f"{seg['id']}: opening {o['from_m']:.3f}-{o['to_m']:.3f} runs past the back face "
                           f"({back_lo:.3f}-{back_hi:.3f}); its jambs would cross a mitred corner")
    return bad
