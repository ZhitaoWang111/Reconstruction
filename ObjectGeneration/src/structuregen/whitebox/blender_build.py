"""Runs inside Blender (headless): build the white model from a structuregen plan.

    blender -b --factory-startup --python src/structuregen/whitebox/blender_build.py -- <plan.json> <out dir>

The mesh of a segment is a lattice of cells along the wall x up its height, one cell
thick. A cell inside an opening is left out and only faces on the outside of the solid
are written, so the body is closed by construction and its faces fall into zones:
front (room side), back, top, bottom, end_start, end_end, reveal (opening jambs and head).

Floors come after the walls: the handoff's inner outline as a slab (top at z = 0, the walls'
base), minus every wall footprint, exact boolean. Each piece left is a room floor, room_1 the
largest. Under each handoff opening that reaches the floor, a threshold fills the gap the cut
left. A floor's faces fall into zones top, bottom, side.

Also writes the handoff's key-frame cameras, a walls.blend, structure_base.usda (root /Structure,
flat; build.py sorts it into Walls/ and Floors/) and a report with every body's closedness and
measured volume, for the caller to compare with the plan.
"""
import json
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
PLAN = json.loads(Path(argv[0]).read_text())
OUT = Path(argv[1])
OUT.mkdir(parents=True, exist_ok=True)
HANDOFF = Path(PLAN["handoff_dir"])


def build(seg):
    """Lattice of cells along x height; faces only on the outside of the solid."""
    ws, we = Vector(seg["front"][0]), Vector(seg["front"][1])
    bs, be = Vector(seg["back"][0]), Vector(seg["back"][1])
    axis = (we - ws)
    u = axis.normalized()
    # Back of the wall: straight behind the front, one thickness away. (bs - ws) is not that: at
    # a mitred start corner it also runs along the wall, and inner lattice columns offset by it
    # would shear every opening's jambs.
    back = (bs - ws) - u * (bs - ws).dot(u)
    normal = back.normalized() * seg["thickness_m"]
    n = normal.normalized()
    H = seg["height_m"]
    opens = seg["openings"]
    al = sorted({seg["a0"], seg["a1"], *[v for o in opens for v in (o["from_m"], o["to_m"])]})
    zs = sorted({0.0, H, *[v for o in opens for v in o["z_floor"]]})

    def point(k, j, side):
        if k == 0:
            p = ws if side == 0 else bs
        elif k == len(al) - 1:
            p = we if side == 0 else be
        else:
            p = ws + u * al[k] + (normal if side else Vector((0, 0)))
        return Vector((p.x, p.y, zs[j]))

    def void(c, l):
        if not (0 <= c < len(al) - 1 and 0 <= l < len(zs) - 1):
            return None
        return any(o["from_m"] <= al[c] and al[c + 1] <= o["to_m"]
                   and o["z_floor"][0] <= zs[l] and zs[l + 1] <= o["z_floor"][1] for o in opens)

    origin = Vector((ws.x, ws.y, 0.0))
    world = Matrix(((u.x, -u.y, 0, origin.x), (u.y, u.x, 0, origin.y), (0, 0, 1, 0), (0, 0, 0, 1)))
    inv = world.inverted()
    index, verts, faces, zones = {}, [], [], []

    def v(k, j, side):
        key = (k, j, side)
        if key not in index:
            index[key] = len(verts)
            verts.append(tuple(inv @ point(k, j, side)))
        return index[key]

    def face(zone, *corners):
        faces.append([v(*c) for c in corners])
        zones.append(zone)

    for c in range(len(al) - 1):
        for l in range(len(zs) - 1):
            if void(c, l):
                continue
            face("front", (c, l, 0), (c + 1, l, 0), (c + 1, l + 1, 0), (c, l + 1, 0))
            face("back", (c, l, 1), (c, l + 1, 1), (c + 1, l + 1, 1), (c + 1, l, 1))
            for zone, neighbour, corners in (
                    ("bottom", void(c, l - 1), ((c, l, 0), (c, l, 1), (c + 1, l, 1), (c + 1, l, 0))),
                    ("top", void(c, l + 1), ((c, l + 1, 0), (c + 1, l + 1, 0), (c + 1, l + 1, 1), (c, l + 1, 1))),
                    ("end_start", void(c - 1, l), ((c, l, 0), (c, l + 1, 0), (c, l + 1, 1), (c, l, 1))),
                    ("end_end", void(c + 1, l), ((c + 1, l, 0), (c + 1, l, 1), (c + 1, l + 1, 1), (c + 1, l + 1, 0)))):
                if neighbour is None:
                    face(zone, *corners)
                elif neighbour:
                    face("reveal", *corners)
    mesh = bpy.data.meshes.new(seg["id"] + "_body")
    mesh.from_pydata(verts, [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    closed = all(len(e.link_faces) == 2 for e in bm.edges)
    volume = bm.calc_volume()
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(seg["id"], mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = world
    obj["handoff_wall"] = seg["wall"]
    obj["along_from_m"] = seg["a0"]
    obj["along_to_m"] = seg["a1"]
    obj["sg_thickness"] = seg["thickness_m"]
    obj["sg_height"] = H
    # openings in segment-local metres (x from this segment's start), for the sg operators
    obj["sg_openings"] = [[o["from_m"] - seg["a0"], o["to_m"] - seg["a0"], o["z_floor"][0], o["z_floor"][1]]
                          for o in opens]
    return obj, zones, closed, volume, len(faces)


def prism(name, pts, z0, z1):
    """A closed prism over a plan polygon, as a new mesh object (world coordinates)."""
    bm = bmesh.new()
    lo = [bm.verts.new((p[0], p[1], z0)) for p in pts]
    hi = [bm.verts.new((p[0], p[1], z1)) for p in pts]
    bm.faces.new(lo[::-1])
    bm.faces.new(hi)
    for k in range(len(pts)):
        bm.faces.new((lo[k], lo[(k + 1) % len(pts)], hi[(k + 1) % len(pts)], hi[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def zone_of(normal):
    return "top" if normal.z > 0.99 else "bottom" if normal.z < -0.99 else "side"


def floor_object(name, bm_faces, kind, extra):
    """One floor body from a set of bmesh faces (copied), zones by normal, white materials."""
    bm = bmesh.new()
    vmap = {}
    for f in bm_faces:
        for v in f.verts:
            if v.index not in vmap:
                vmap[v.index] = bm.verts.new(v.co)
    zones = []
    for f in bm_faces:
        bm.faces.new([vmap[v.index] for v in f.verts])
        zones.append(zone_of(f.normal))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    closed = all(len(e.link_faces) == 2 for e in bm.edges)
    volume = bm.calc_volume()
    tops = [[list(v.co[:2]) for v in f.verts] for f in bm.faces if f.normal.z > 0.99]
    area = sum(f.calc_area() for f in bm.faces if f.normal.z > 0.99)
    me = bpy.data.meshes.new(name + "_body")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob["sg_floor"] = kind
    ob["sg_thickness"] = FLOOR_T
    for k, v in extra.items():
        ob[k] = v
    mats = {z: material(f"{name}_{z}") for z in sorted(set(zones))}
    for m in mats.values():
        me.materials.append(m)
    names = [m.name for m in me.materials]
    for poly, z in zip(me.polygons, zones):
        poly.material_index = names.index(mats[z].name)
    return {"id": name, "kind": kind, "closed": closed, "volume_m3": round(volume, 6),
            "area_m2": round(area, 6), "top_polygons": [[[round(c, 5) for c in p] for p in poly] for poly in tops],
            "faces": len(zones), "zones": sorted(set(zones)), **extra}


def build_floors(outline):
    """Room floors: the outline slab minus every wall footprint; thresholds under floor-level openings."""
    slab = prism("_slab", outline, -FLOOR_T, 0.0)
    for seg in PLAN["segments"]:
        cut = prism("_cut", [seg["front"][0], seg["front"][1], seg["back"][1], seg["back"][0]], -1.0, 1.0)
        mod = slab.modifiers.new("cut", "BOOLEAN")
        mod.operation, mod.solver, mod.object = "DIFFERENCE", "EXACT", cut
        bpy.context.view_layer.objects.active = slab
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cut)
    bm = bmesh.new()
    bm.from_mesh(slab.data)
    bmesh.ops.dissolve_limit(bm, angle_limit=0.0005, verts=bm.verts[:], edges=bm.edges[:])
    bm.verts.index_update()
    seen, parts = set(), []
    for f in bm.faces:
        if f.index in seen:
            continue
        stack, comp = [f], []
        seen.add(f.index)
        while stack:
            g = stack.pop()
            comp.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
        parts.append(comp)
    parts.sort(key=lambda c: -sum(f.calc_area() for f in c if f.normal.z > 0.99))
    floors = []
    for i, comp in enumerate(parts, 1):
        floors.append(floor_object(f"room_{i}", comp, "room", {}))
    bm.free()
    bpy.data.objects.remove(slab)

    for seg in PLAN["segments"]:
        ws, we = Vector(seg["front"][0]), Vector(seg["front"][1])
        u = (we - ws).normalized()
        b = Vector(seg["back"][0]) - ws
        n = (b - u * b.dot(u)).normalized() * seg["thickness_m"]
        for k, o in enumerate(seg["openings"]):
            if o["z_floor"][0] > 1e-6:
                continue
            p0, p1 = ws + u * (o["from_m"] - seg["a0"]), ws + u * (o["to_m"] - seg["a0"])
            ob = prism("_thr", [p0, p1, p1 + n, p0 + n], -FLOOR_T, 0.0)
            tbm = bmesh.new()
            tbm.from_mesh(ob.data)
            tbm.verts.index_update()
            name = f"threshold_{seg['id']}_{k}"
            floors.append(floor_object(name, list(tbm.faces), "threshold", {"sg_opening_of": seg["id"], "sg_opening": k}))
            tbm.free()
            bpy.data.objects.remove(ob)
    return floors


def material(name):
    """One white placeholder for every zone; the appearance stage replaces these."""
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    return mat


def add_camera(doc, fid, c2w):
    """handoff camera_to_world is OpenCV (+Z forward, +Y down); a Blender camera looks -Z with +Y up."""
    cam = doc["camera"]
    data = bpy.data.cameras.new(f"frame_{fid:04d}")
    data.sensor_fit, data.sensor_width = "HORIZONTAL", 36.0
    data.lens = cam["fx"] * 36.0 / cam["width"]
    data.shift_x = -(cam["cx"] - cam["width"] / 2) / cam["width"]
    data.shift_y = (cam["cy"] - cam["height"] / 2) / cam["width"]
    data.clip_start, data.clip_end = 0.05, 50.0
    ob = bpy.data.objects.new(f"frame_{fid:04d}", data)
    bpy.context.scene.collection.objects.link(ob)
    ob.matrix_world = Matrix(c2w) @ Matrix.Diagonal((1, -1, -1, 1))
    return ob


for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)

report = {"schema": "structuregen/build/v2", "blender": bpy.app.version_string, "segments": [], "floors": [],
          "frames": []}
FLOOR_T = float(PLAN.get("floor_thickness_m", 0.05))
# No parent empty: the exporter's root_prim_path already gives /Walls, and segments sit
# directly under it as /Walls/<segment> with their world transform.
for seg in PLAN["segments"]:
    obj, zones, closed, volume, face_count = build(seg)
    mats = {z: material(f"{seg['id']}_{z}") for z in sorted(set(zones))}
    for mat in mats.values():
        obj.data.materials.append(mat)
    names = [m.name for m in obj.data.materials]
    for poly, z in zip(obj.data.polygons, zones):
        poly.material_index = names.index(mats[z].name)
    report["segments"].append({
        "id": seg["id"], "wall": seg["wall"], "closed": closed,
        "volume_m3": round(volume, 6), "expected_volume_m3": round(seg["expected_volume_m3"], 6),
        "volume_error_m3": round(volume - seg["expected_volume_m3"], 6),
        "faces": face_count, "zones": sorted(set(zones)),
    })

doc = json.loads((HANDOFF / "walls.json").read_text())
outline = doc.get("outline", {}).get("inner_corners_floor")
if outline:
    report["floors"] = build_floors(outline)
seen = {}
for w in doc["walls"]:
    for k in w["key_frames"]:
        seen.setdefault(k["source_frame_id"], k["camera_to_world"])
for fid, c2w in sorted(seen.items()):
    add_camera(doc, fid, c2w)
    report["frames"].append(fid)

bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "walls.blend"))
for ob in list(bpy.data.objects):
    if ob.type == "CAMERA":
        bpy.data.objects.remove(ob)
bpy.ops.wm.usd_export(filepath=str(OUT / "structure_base.usda"), root_prim_path="/Structure",
                      export_materials=True, generate_preview_surface=True, export_cameras=False,
                      export_lights=False, export_custom_properties=True,
                      convert_world_material=False, export_uvmaps=False, xform_op_mode="MAT")
(OUT / "build_report.json").write_text(json.dumps(report, indent=1))
print("STRUCTUREGEN_BUILT", len(report["segments"]), "segments,", len(report["floors"]), "floors,",
      len(report["frames"]), "cameras")
