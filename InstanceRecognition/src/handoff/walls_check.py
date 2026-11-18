"""Check a wall handoff directory (walls/v2) against its contract, whoever produced it.

    .venv/bin/python -m src.handoff.walls_check artifacts/handoff/scene0000_00/walls_harness

walls/v2.

Every check is a promise the README makes to the wall generator:

* the frame (amb3r_floor, with a valid `frame` block) and camera conventions match the object handoff;
* every wall is a straight, non-degenerate line with the room on its left, a unit
  `normal_to_back` pointing away from the room, a positive thickness and height;
* `segments` run end to end along the wall, without gaps or overlaps;
* `joins` name walls that exist, and a joined end really touches the wall it names;
* the exterior walls close: each one ends where the next begins, inside and out;
* `footprint_floor` is what the line, the thickness and the joins say it is;
* key frames: at most three, distinct roles, a rigid camera_to_world, the image files
  exist, the mask's pixel count and box agree with the record, `covers_m` lies on the wall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import schema, walls_schema as ws
from .check import frame_problems

TOL = 1e-3          # m; geometric agreement
JOIN_TOL = 0.02     # m; a joined end must lie this close to the other wall's line


def _rigid(pose) -> bool:
    T = np.asarray(pose, dtype=np.float64)
    return (T.shape == (4, 4) and np.isfinite(T).all() and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-3) and np.linalg.det(T[:3, :3]) > 0)


def _dist_to_line(p, a, b) -> float:
    d = (b - a) / np.linalg.norm(b - a)
    w = p - a
    return float(abs(w[0] * d[1] - w[1] * d[0]))


def check(directory: Path) -> list[str]:
    problems: list[str] = []
    path = directory / "walls.json"
    if not path.is_file():
        return ["missing walls.json"]
    doc = json.loads(path.read_text())

    # ---- top level
    if doc.get("schema_version") != ws.SCHEMA_VERSION:
        problems.append(f"schema_version {doc.get('schema_version')!r} != {ws.SCHEMA_VERSION!r}")
    missing = [k for k in ws.TOP_REQUIRED if k not in doc]
    if missing:
        return problems + [f"missing top-level fields {missing}"]
    if (doc["coordinate_frame"], doc["units"], doc["up_axis"]) != (ws.FRAME, "meters", "Z"):
        problems.append(f"coordinate_frame/units/up_axis differ from the object handoff ({ws.FRAME}, meters, Z)")
    problems += frame_problems(doc["frame"])
    cam = doc["camera"]
    cam_missing = [k for k in schema.CAMERA_REQUIRED if k not in cam]
    if cam_missing:
        problems.append(f"camera missing {cam_missing}")
    else:
        K = np.asarray(cam["K"], dtype=np.float64)
        if not np.allclose(K, [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]]):
            problems.append("camera K disagrees with fx/fy/cx/cy")
        if cam["extrinsics"] != schema.EXTRINSICS_CONVENTION:
            problems.append("camera extrinsics convention differs from the object handoff")
    if doc["storey"].get("kind") not in ws.VALUE_KINDS or not doc["storey"].get("height_m", 0) > 0:
        problems.append("storey needs a positive height_m and kind measured|assumed")
    if doc["thickness_defaults"].get("kind") not in ws.VALUE_KINDS:
        problems.append("thickness_defaults.kind must be measured|assumed")
    if set(doc["not_included"]) != set(ws.NOT_INCLUDED):
        problems.append(f"not_included {doc['not_included']} != {list(ws.NOT_INCLUDED)}")

    walls = doc["walls"]
    by_id = {}
    for w in walls:
        if w.get("id") in by_id:
            problems.append(f"duplicate wall id {w.get('id')}")
        by_id[w.get("id")] = w

    # ---- each wall
    for w in walls:
        name = w.get("id", "<no id>")
        miss = [k for k in ws.WALL_REQUIRED if k not in w]
        if miss:
            problems.append(f"{name}: missing {miss}")
            continue
        if w["role"] not in ws.ROLES:
            problems.append(f"{name}: role {w['role']!r}")
        a, b = np.asarray(w["start_floor"], float), np.asarray(w["end_floor"], float)
        L = float(np.linalg.norm(b - a))
        if L < 0.05:
            problems.append(f"{name}: degenerate wall line ({L:.3f} m)"); continue
        if abs(L - w["length_m"]) > TOL:
            problems.append(f"{name}: length_m {w['length_m']} != |end - start| {L:.4f}")
        d = (b - a) / L
        back = np.asarray(w["normal_to_back"], float)
        if abs(np.linalg.norm(back) - 1) > 1e-4 or abs(back @ d) > 1e-3:
            problems.append(f"{name}: normal_to_back is not a unit vector perpendicular to the wall")
        if back @ np.array([-d[1], d[0]]) > -0.999:
            problems.append(f"{name}: the room is not on the left of start->end (normal_to_back must point right)")
        if not w["thickness_m"] > 0 or w["thickness_kind"] not in ws.VALUE_KINDS:
            problems.append(f"{name}: thickness_m must be positive with kind measured|assumed")
        if not w["height_m"] > 0:
            problems.append(f"{name}: height_m must be positive")
        # openings: measured holes, each inside the wall, on the wall's line and height
        spans = []
        for o in w["openings"]:
            miss = [k for k in ws.OPENING_REQUIRED if k not in o]
            if miss:
                problems.append(f"{name}: opening missing {miss}"); continue
            tag = f"{name} opening {o['from_m']}-{o['to_m']} m"
            if not 0 <= o["from_m"] < o["to_m"] <= L + TOL:
                problems.append(f"{tag}: outside the wall's 0..{L:.3f} m")
            if o["z_floor"][0] < -TOL or o["z_floor"][0] >= o["z_floor"][1] or o["z_floor"][1] > w["height_m"] + TOL:
                problems.append(f"{tag}: z_floor {o['z_floor']} not inside 0..{w['height_m']}")
            if abs(o["width_m"] - (o["to_m"] - o["from_m"])) > TOL:
                problems.append(f"{tag}: width_m disagrees with from_m/to_m")
            if abs(o["height_m"] - (o["z_floor"][1] - o["z_floor"][0])) > TOL:
                problems.append(f"{tag}: height_m disagrees with z_floor")
            edges = o["bounds"] or {}
            if any(edges.get(k) not in ws.OPENING_SIDE_BOUNDS for k in ("left", "right", "top")) \
                    or edges.get("bottom") not in ws.OPENING_BOTTOM_BOUNDS:
                problems.append(f"{tag}: bounds {edges} - sides and top must be solid, bottom solid or floor")
            elif (edges["bottom"] == "floor") != (abs(o["z_floor"][0]) <= TOL):
                problems.append(f"{tag}: bottom bound {edges['bottom']} disagrees with z_floor {o['z_floor']}")
            if o["kind"] != "opening" or o["source"] not in ws.OPENING_SOURCES:
                problems.append(f"{tag}: kind must be opening, source one of {ws.OPENING_SOURCES}")
            quad = np.asarray(o["corners_floor"], float)
            if quad.shape != (4, 3):
                problems.append(f"{tag}: corners_floor must be four [x, y, z] points")
            else:
                for q in quad:
                    if _dist_to_line(q[:2], a, b) > JOIN_TOL or q[2] < -TOL or q[2] > w["height_m"] + TOL:
                        problems.append(f"{tag}: corner {q.tolist()} is not on the wall's face"); break
                expect = [[*(a + s_ * d), z] for s_, z in ((o["from_m"], o["z_floor"][0]), (o["to_m"], o["z_floor"][0]),
                                                           (o["to_m"], o["z_floor"][1]), (o["from_m"], o["z_floor"][1]))]
                if np.abs(quad - np.asarray(expect)).max() > 0.002:
                    problems.append(f"{tag}: corners_floor disagree with from_m/to_m/z_floor")
            spans.append((o["from_m"], o["to_m"]))
        spans.sort()
        for (a0, a1), (b0, _) in zip(spans, spans[1:]):
            if b0 < a1 - TOL:
                problems.append(f"{name}: openings {a0}-{a1} and {b0}- overlap")

        segs = w["segments"]
        if not segs:
            problems.append(f"{name}: no segments")
        else:
            if abs(segs[0]["from_m"]) > TOL or abs(segs[-1]["to_m"] - L) > TOL:
                problems.append(f"{name}: segments run {segs[0]['from_m']}..{segs[-1]['to_m']}, not 0..{L:.3f}")
            for s0, s1 in zip(segs, segs[1:]):
                if abs(s0["to_m"] - s1["from_m"]) > TOL:
                    problems.append(f"{name}: segments leave a gap or overlap at {s0['to_m']}")
            for s in segs:
                if [k for k in ws.SEGMENT_REQUIRED if k not in s] or s["kind"] not in ws.SEGMENT_KINDS or s["to_m"] <= s["from_m"]:
                    problems.append(f"{name}: bad segment {s}")
                elif s["kind"] == "observed" and not s["source_planes"]:
                    problems.append(f"{name}: observed segment without source_planes")

        # joins: the named wall exists and this end touches its line
        for side, p in (("start", a), ("end", b)):
            j = w["joins"].get(side) or {}
            if j.get("type") not in ws.JOIN_TYPES:
                problems.append(f"{name}: joins.{side}.type {j.get('type')!r}"); continue
            if j["type"] == "none":
                continue
            o = by_id.get(j.get("wall"))
            if o is None:
                problems.append(f"{name}: joins.{side} names unknown wall {j.get('wall')!r}"); continue
            dist = _dist_to_line(p, np.asarray(o["start_floor"], float), np.asarray(o["end_floor"], float))
            if dist > JOIN_TOL:
                problems.append(f"{name}: {side} is {dist * 100:.1f} cm off {o['id']}, which it joins")

        # footprint: the room face is the line, the back face is `thickness` behind it
        fp = np.asarray(w["footprint_floor"], float)
        if fp.shape != (4, 2):
            problems.append(f"{name}: footprint_floor must be four [x, y] points")
        else:
            if np.abs(fp[0] - a).max() > TOL or np.abs(fp[1] - b).max() > TOL:
                problems.append(f"{name}: footprint does not start with the wall line")
            for q in fp[2:]:
                if abs((q - a) @ back - w["thickness_m"]) > TOL:
                    problems.append(f"{name}: footprint back corner is not thickness_m behind the line")

        # key frames
        kfs = w["key_frames"]
        if len(kfs) > ws.MAX_KEYFRAMES:
            problems.append(f"{name}: {len(kfs)} key frames, max {ws.MAX_KEYFRAMES}")
        roles = [k.get("role") for k in kfs]
        if roles != list(schema.KEYFRAME_ROLES[:len(roles)]):
            problems.append(f"{name}: key frame roles {roles} are not primary, secondary, third in order")
        for k in kfs:
            tag = f"{name} frame {k.get('source_frame_id')}"
            miss = [x for x in ws.KEYFRAME_REQUIRED if x not in k]
            if miss:
                problems.append(f"{tag}: missing {miss}"); continue
            if not _rigid(k["camera_to_world"]):
                problems.append(f"{tag}: camera_to_world is not a rigid 4x4")
            if k["bbox_convention"] != schema.BBOX_CONVENTION:
                problems.append(f"{tag}: bbox_convention differs")
            c0, c1 = k["covers_m"]
            if not (-TOL <= c0 <= c1 <= L + TOL):
                problems.append(f"{tag}: covers_m {k['covers_m']} is not within 0..{L:.3f}")
            for key in ("rgb", "mask", "boxed"):
                if not (directory / k[key]).is_file():
                    problems.append(f"{tag}: {key} image missing")
            if (directory / k["mask"]).is_file():
                m = np.asarray(Image.open(directory / k["mask"])) > 0
                if (m.shape[1], m.shape[0]) != (cam.get("width"), cam.get("height")):
                    problems.append(f"{tag}: mask is {m.shape[1]}x{m.shape[0]}, camera is {cam.get('width')}x{cam.get('height')}")
                if int(m.sum()) != k["visible_pixels"]:
                    problems.append(f"{tag}: visible_pixels {k['visible_pixels']} != {int(m.sum())} set mask pixels")
                ys, xs = np.nonzero(m)
                if len(xs) and [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] != k["bbox_xyxy"]:
                    problems.append(f"{tag}: bbox_xyxy disagrees with the mask")
            r = k.get("rectified")
            if r is not None and ([x for x in ws.RECTIFIED_REQUIRED if x not in r] or not (directory / r["image"]).is_file()):
                problems.append(f"{tag}: rectified view incomplete")
        if not kfs and any(s["kind"] == "observed" for s in segs):
            problems.append(f"{name}: observed but has no key frame")

    # ---- the exterior walls close, inside and out
    ext = [w for w in walls if w.get("role") == "exterior"]
    for w in ext:
        nxt = by_id.get((w["joins"].get("end") or {}).get("wall"))
        if nxt is None or nxt.get("role") != "exterior":
            problems.append(f"{w['id']}: end does not join another exterior wall"); continue
        if np.abs(np.asarray(w["end_floor"]) - np.asarray(nxt["start_floor"])).max() > TOL:
            problems.append(f"{w['id']} -> {nxt['id']}: room-side corner is open")
        if np.abs(np.asarray(w["footprint_floor"][2]) - np.asarray(nxt["footprint_floor"][3])).max() > TOL:
            problems.append(f"{w['id']} -> {nxt['id']}: outside corner is open")
    if ext:
        loop, cur = [], ext[0]
        while cur is not None and cur["id"] not in loop:
            loop.append(cur["id"]); cur = by_id.get((cur["joins"].get("end") or {}).get("wall"))
        if len(loop) != len(ext) or cur is None or cur["id"] != loop[0]:
            problems.append(f"exterior walls do not form one closed loop ({loop})")
        corners = np.asarray(doc["outline"]["inner_corners_floor"], float)
        area = 0.5 * float(np.sum(corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1]))
        if area <= 0:
            problems.append("outline.inner_corners_floor is not counter-clockwise")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", type=Path)
    problems = check(ap.parse_args().directory)
    for p in problems:
        print(f"FAIL {p}")
    if problems:
        raise SystemExit(1)
    print(f"OK {ap.parse_args().directory}")


if __name__ == "__main__":
    main()
