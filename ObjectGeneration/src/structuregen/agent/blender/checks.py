"""Automatic checks, run inside Blender after every edit and again at finish_segment.

Loaded by `exec` next to sg_ops.py. `reference()` records the white model once; `check(...)`
compares the live scene with it and returns a dict the harness turns into the agent's report:

    hard     problems that roll the edit back: wall geometry changed, another segment touched,
             an object that is not a segment body, a camera or an sg trim
    trims    problems kept but reported, blocking finish: a trim off the room-side face, past the
             segment, through an opening, inside another wall, or outside its size limits
    faces    room-side faces still without a colour set by sg.paint (progress; blocks finish)
    items    which declared items have something done for them (progress; blocks finish)
"""
import hashlib

import bpy
from mathutils import Vector

LIMITS = {"depth": (0.003, 0.05), "skirting_h": (0.02, 0.30), "rail_h": (0.01, 0.30), "casing_w": (0.02, 0.20)}
EPS = 0.002


def _mesh_hash(ob):
    h = hashlib.sha1()
    for v in ob.data.vertices:
        h.update(("%.5f,%.5f,%.5f;" % tuple(v.co)).encode())
    for p in ob.data.polygons:
        h.update((",".join(map(str, p.vertices)) + "|%d;" % p.material_index).encode())
    m = ob.matrix_world
    h.update(("%.5f" * 16 % tuple(m[i][j] for i in range(4) for j in range(4))).encode())
    return h.hexdigest()


def _material_state(mat):
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mat.node_tree else None
    if bsdf is None:
        return None
    tiles = mat.get("sg_tiles")
    return [round(float(c), 4) for c in bsdf.inputs["Base Color"].default_value[:3]] + \
        [round(float(bsdf.inputs["Roughness"].default_value), 3), tiles]


def body_map():
    """Every body the white model built: wall segments and floors."""
    return {o.name: o for o in bpy.data.objects
            if o.type == "MESH" and o.parent is None and ("handoff_wall" in o or "sg_floor" in o)}


def _is_floor(ob):
    return "sg_floor" in ob


def reference():
    """Fingerprint of the white model, before the agent edits anything."""
    return {"bodies": {n: _mesh_hash(o) for n, o in body_map().items()},
            "cameras": sorted(o.name for o in bpy.data.objects if o.type == "CAMERA")}


def snapshot():
    """Everything an edit can change, per segment: body mesh, its materials, its trims."""
    state = {}
    for name, ob in body_map().items():
        mats = {m.name: _material_state(m) for m in ob.data.materials if m}
        trims = {}
        for c in ob.children:
            trims[c.name] = [_mesh_hash(c) if c.type == "MESH" else c.type,
                             [_material_state(m) for m in getattr(c.data, "materials", []) if m]]
        state[name] = {"body": _mesh_hash(ob), "materials": mats, "trims": trims}
    return state


def _segment_problems(seg):
    """Trim problems for one segment (segment-local frame: X along, Y into the room, Z up)."""
    out = []
    length = float(seg["along_to_m"]) - float(seg["along_from_m"])
    height = float(seg["sg_height"])
    thick = float(seg["sg_thickness"])
    openings = [tuple(o) for o in seg.get("sg_openings", [])]
    for c in seg.children:
        kind = c.get("sg")
        if c.type != "MESH" or kind not in ("skirting", "rail", "casing"):
            continue
        parts = list(c.get("sg_parts", []))
        boxes = [parts[i:i + 6] for i in range(0, len(parts), 6)]
        tag = f"{c.name}"
        for x0, x1, y0, y1, z0, z1 in boxes:
            depth = y1 - y0
            front = y0 >= -EPS
            back = y1 <= -thick + EPS
            if not (front or back):
                out.append(f"{tag}: not on a face of the wall (y {y0:.3f}..{y1:.3f})")
            if not LIMITS["depth"][0] - 1e-6 <= depth <= LIMITS["depth"][1] + 1e-6:
                out.append(f"{tag}: sticks out {depth * 100:.1f} cm (allowed {LIMITS['depth'][0] * 100:.1f}–{LIMITS['depth'][1] * 100:.0f} cm)")
            if z0 < -EPS or z1 > height + EPS:
                out.append(f"{tag}: height {z0:.3f}–{z1:.3f} m outside the wall (0–{height:.2f} m)")
            if kind != "casing":
                if x0 < -EPS or x1 > length + EPS:
                    out.append(f"{tag}: runs {x0:.3f}–{x1:.3f} m, past this segment (0–{length:.3f} m)")
                for a0, a1, oz0, oz1 in openings:
                    if x0 < a1 - EPS and x1 > a0 + EPS and z0 < oz1 - EPS and z1 > oz0 + EPS:
                        out.append(f"{tag}: crosses the opening at {a0:.2f}–{a1:.2f} m")
        if kind in ("skirting", "rail") and boxes:
            h = boxes[0][5] - boxes[0][4]
            lo, hi = LIMITS[f"{kind}_h"]
            if not lo - 1e-6 <= h <= hi + 1e-6:
                out.append(f"{tag}: {kind} {h * 100:.1f} cm high (allowed {lo * 100:.0f}–{hi * 100:.0f} cm)")
        if kind == "casing":
            w = float(c.get("sg_width", 0))
            if not LIMITS["casing_w"][0] - 1e-6 <= w <= LIMITS["casing_w"][1] + 1e-6:
                out.append(f"{tag}: casing {w * 100:.1f} cm wide (allowed {LIMITS['casing_w'][0] * 100:.0f}–{LIMITS['casing_w'][1] * 100:.0f} cm)")
        # inside another wall: sample each box's room-side face
        samples = []
        for x0, x1, y0, y1, z0, z1 in boxes:
            n = max(2, int((x1 - x0) / 0.02) + 1)            # every 2 cm along, ends included
            for i in range(n):
                x = x0 + (x1 - x0) * i / (n - 1)
                x = min(max(x, x0 + 0.004), x1 - 0.004)      # just inside the box's own ends
                for fz in (0.25, 0.75):
                    samples.append(seg.matrix_world @ Vector((x, (y0 + y1) / 2, z0 + (z1 - z0) * fz)))
        hits = {h for h in inside_any(samples, exclude=(seg.name,)) if h}
        if hits:
            out.append(f"{tag}: runs into {', '.join(sorted(hits))}")
    return out


def check(ref, before, current_segment, keep=()):
    """Compare the live scene with the white-model reference and with the state before this edit."""
    hard, trims = [], []
    now_bodies = body_map()
    for name, digest in ref["bodies"].items():
        ob = now_bodies.get(name)
        if ob is None:
            hard.append(f"body {name} was deleted")
        elif _mesh_hash(ob) != digest:
            hard.append(f"{'floor' if _is_floor(ob) else 'wall'} geometry of {name} changed (vertices, faces or placement)")
    cams = sorted(o.name for o in bpy.data.objects if o.type == "CAMERA")
    if cams != ref["cameras"]:
        hard.append("key-frame cameras were changed")
    for ob in bpy.data.objects:
        if ob.type == "CAMERA" or ob.name in now_bodies:
            continue
        if not (ob.get("sg") in ("skirting", "rail", "casing") and ob.parent is not None and ob.parent.name in now_bodies):
            hard.append(f"object {ob.name!r} was not made by sg.strip / sg.frame")
    after = snapshot()
    allowed = [current_segment] if isinstance(current_segment, str) else list(current_segment)
    if before is not None:
        for name in ref["bodies"]:
            if name not in allowed and after.get(name) != before.get(name):
                hard.append(f"segment {name} changed, but only {', '.join(allowed)} may be edited now")
    if before is not None:
        for seg_name, mat_name in keep:           # finished room-side faces and trims stay as they are
            b, a = before.get(seg_name, {}), after.get(seg_name, {})
            if b.get("materials", {}).get(mat_name) != a.get("materials", {}).get(mat_name):
                hard.append(f"{mat_name} was finished with its wall and must not change now")
        for seg_name in sorted({k[0] for k in keep}):
            if before.get(seg_name, {}).get("trims") != after.get(seg_name, {}).get("trims"):
                hard.append(f"trims of {seg_name} were finished with its wall and must not change now")
    for name in allowed:
        seg = now_bodies.get(name)
        if seg is not None and not _is_floor(seg):
            trims += _segment_problems(seg)
    return {"hard": hard, "trims": trims, "state": after}


def progress(segment, items, room_faces):
    """Faces not yet painted and declared items with nothing done for them, for one segment."""
    seg = body_map()[segment]
    unpainted = []
    for face in room_faces:
        mat = bpy.data.materials.get(f"{segment}_{face}")
        if mat is None or "sg_rgb" not in mat:
            unpainted.append(face)
    done = set()
    for m in seg.data.materials:
        if m and m.get("sg_item"):
            done.add(m["sg_item"])
    for c in seg.children:
        if c.get("sg_item"):
            done.add(c["sg_item"])
    return {"unpainted": unpainted, "items_done": sorted(i for i in items if i in done),
            "items_open": sorted(i for i in items if i not in done)}
