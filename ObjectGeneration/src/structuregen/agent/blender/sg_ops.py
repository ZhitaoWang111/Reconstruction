"""The operators the appearance agent edits walls and floors with, registered inside Blender.

    bpy.ops.sg.paint(seg="ext_E_1", face="front", rgb=(199, 190, 176), roughness=0.9, item="t1")
    bpy.ops.sg.tiles(seg="room_1", face="top", size=(0.30, 0.30), joint=0.003, rgb=(...), joint_rgb=(...),
                     angle=0, offset=0.0, roughness=0.6, item="t1")
    bpy.ops.sg.strip(seg="ext_E_1", z0=0.96, z1=1.14, a0=-1, a1=-1, depth=0.02, rgb=(...), kind="rail", item="t2")
    bpy.ops.sg.frame(seg="int_3", width=0.06, depth=0.02, sides="FRONT", rgb=(...), item="t3")

Loaded by `exec` from the harness (no module import: MCP safe mode only allows `bpy.ops.*` calls
in agent code, so these are operators, not functions). Everything is in a segment's local frame,
the one the white model was built in: X along the wall from the segment's front start corner,
Y into the room, Z up, metres. Lengths along the wall here are segment-local (0 .. segment length).

Each operator only writes what it is told to; checks live in the harness (`checks.py`). Trims
are children of their segment, named `<seg>__<kind>__<item>`, so one `item` owns one object
and calling again with the same item replaces it.
"""
import json

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SG_PROP = "sg"            # every object this module creates carries obj["sg"] = kind
TRIM_KINDS = ("skirting", "rail", "casing")


def _linear(c):
    c = max(0.0, min(255.0, float(c))) / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _material(name, rgb, roughness):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    lin = [_linear(c) for c in rgb] + [1.0]
    bsdf.inputs["Base Color"].default_value = lin
    bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = lin
    mat["sg_rgb"] = [int(round(c)) for c in rgb]
    return mat


def _segment(name):
    ob = bpy.data.objects.get(name)
    if ob is None or ob.type != "MESH" or ob.parent is not None or "handoff_wall" not in ob:
        raise ValueError(f"no wall segment named {name!r}")
    return ob


def _body(name):
    """A wall segment or a floor: the bodies sg.paint and sg.tiles may colour."""
    ob = bpy.data.objects.get(name)
    if ob is None or ob.type != "MESH" or ob.parent is not None or not ("handoff_wall" in ob or "sg_floor" in ob):
        raise ValueError(f"no wall segment or floor named {name!r}")
    return ob


def _plain(mat):
    """Back to a single base colour: drop the tile image of an earlier sg.tiles."""
    nodes = mat.node_tree.nodes
    for n in [n for n in nodes if n.type == "TEX_IMAGE"]:
        img = n.image
        nodes.remove(n)
        if img is not None and img.users == 0:
            bpy.data.images.remove(img)
    for k in ("sg_tiles",):
        if k in mat:
            del mat[k]


def _length(seg):
    return float(seg["along_to_m"]) - float(seg["along_from_m"])


def bodies():
    """Every wall segment body (top-level meshes the white model built)."""
    return [o for o in bpy.data.objects if o.type == "MESH" and o.parent is None and "handoff_wall" in o]


def _inside(tree, q, tol):
    """Ray parity in the body's local frame: inside a closed mesh, a ray crosses it an odd number
    of times. A point within tol of the surface counts as outside (touching is allowed)."""
    loc, _, _, dist = tree.find_nearest(q)
    if loc is None or dist <= tol:
        return False
    d = Vector((0.5773, 0.5774, 0.5773))          # off-axis, so the ray never runs along a face
    crossings, origin = 0, q.copy()
    for _ in range(64):
        hit, _, _, _ = tree.ray_cast(origin, d)
        if hit is None:
            break
        crossings += 1
        origin = hit + d * 1e-5
    return crossings % 2 == 1


def inside_any(points_world, exclude=(), tol=0.002):
    """For each world point: the name of the wall body it sits inside (deeper than tol), or None."""
    deps = bpy.context.evaluated_depsgraph_get()
    # FromObject builds the tree in the object's local frame: query with local points
    trees = []
    for o in bodies():
        if o.name in exclude:
            continue
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        lo = Vector([min(v[i] for v in bb) for i in range(3)])
        hi = Vector([max(v[i] for v in bb) for i in range(3)])
        trees.append((o.name, BVHTree.FromObject(o, deps), o.matrix_world.inverted(), lo, hi))
    out = []
    for p in points_world:
        hit = None
        for name, tree, inv, lo, hi in trees:
            if any(p[i] < lo[i] - tol or p[i] > hi[i] + tol for i in range(3)):
                continue
            if _inside(tree, inv @ p, tol):
                hit = name
                break
        out.append(hit)
    return out


def free_span(seg, x0, x1, y, z, step=0.01):
    """Shrink [x0, x1] from both ends while a point at (x, y, z) in the segment's frame lies inside
    another wall (a wall butting onto this one, or a sharp corner). Segment-local metres."""
    n = max(2, int(round((x1 - x0) / step)) + 1)
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    hits = inside_any([seg.matrix_world @ Vector((x, y, z)) for x in xs], exclude=(seg.name,))
    free = [x for x, h in zip(xs, hits) if h is None]
    if not free:
        return x0, x0
    lo, hi = free[0], free[-1]
    return (x0 if lo == xs[0] else lo), (x1 if hi == xs[-1] else hi)


def _openings(seg):
    """Openings of this segment in segment-local metres: [(a0, a1, z0, z1), ...]."""
    return [tuple(o) for o in seg.get("sg_openings", [])]


def _box(name, parent, parts, mat, kind, item, extra=None):
    """Replace the trim named `name` with one mesh of axis-aligned boxes (segment-local)."""
    old = bpy.data.objects.get(name)
    if old is not None:
        mesh = old.data
        bpy.data.objects.remove(old)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    bm = bmesh.new()
    for (x0, x1, y0, y1, z0, z1) in parts:
        if x1 - x0 <= 1e-4 or z1 - z0 <= 1e-4:
            continue
        res = bmesh.ops.create_cube(bm, size=1.0)
        for v in res["verts"]:
            v.co = Vector((x0 if v.co.x < 0 else x1, y0 if v.co.y < 0 else y1, z0 if v.co.z < 0 else z1))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mat)
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.matrix_parent_inverse.identity()      # children sit in the segment's local frame
    ob[SG_PROP] = kind
    ob["sg_item"] = item
    ob["sg_parts"] = [v for part in parts for v in part]      # flattened (x0,x1,y0,y1,z0,z1)*n
    for k, v in (extra or {}).items():
        ob[k] = v
    return ob


def _cut_by_openings(seg, x0, x1, z0, z1):
    """[x0, x1] minus every opening the band [z0, z1] runs into."""
    spans = [(x0, x1)]
    for a0, a1, oz0, oz1 in _openings(seg):
        if oz1 <= z0 or oz0 >= z1:
            continue
        nxt = []
        for s0, s1 in spans:
            if a1 <= s0 or a0 >= s1:
                nxt.append((s0, s1))
                continue
            if a0 > s0:
                nxt.append((s0, a0))
            if a1 < s1:
                nxt.append((a1, s1))
        spans = nxt
    return spans


class SG_OT_paint(bpy.types.Operator):
    """Set the colour and roughness of one face zone of a wall segment"""
    bl_idname = "sg.paint"
    bl_label = "Paint wall face"
    bl_options = {"REGISTER", "UNDO"}
    seg: bpy.props.StringProperty()
    face: bpy.props.StringProperty(default="front")
    rgb: bpy.props.FloatVectorProperty(size=3, default=(200, 200, 200))
    roughness: bpy.props.FloatProperty(default=0.9, min=0.0, max=1.0)
    item: bpy.props.StringProperty()

    def execute(self, context):
        seg = _body(self.seg)
        name = f"{seg.name}_{self.face}"
        if bpy.data.materials.get(name) is None or name not in [m.name for m in seg.data.materials if m]:
            raise ValueError(f"{seg.name} has no face {self.face!r}")
        mat = _material(name, self.rgb, self.roughness)
        _plain(mat)
        mat["sg_item"] = self.item
        print(f"SG paint {name} rgb={tuple(int(round(c)) for c in self.rgb)} item={self.item}")
        return {"FINISHED"}


class SG_OT_strip(bpy.types.Operator):
    """Add a horizontal strip on a segment's room-side face: skirting or rail"""
    bl_idname = "sg.strip"
    bl_label = "Wall strip"
    bl_options = {"REGISTER", "UNDO"}
    seg: bpy.props.StringProperty()
    z0: bpy.props.FloatProperty()
    z1: bpy.props.FloatProperty()
    a0: bpy.props.FloatProperty(default=-1.0)      # -1: from the segment's start
    a1: bpy.props.FloatProperty(default=-1.0)      # -1: to the segment's end
    depth: bpy.props.FloatProperty(default=0.015)
    rgb: bpy.props.FloatVectorProperty(size=3, default=(200, 200, 200))
    roughness: bpy.props.FloatProperty(default=0.6, min=0.0, max=1.0)
    kind: bpy.props.StringProperty(default="skirting")
    item: bpy.props.StringProperty()

    def execute(self, context):
        seg = _segment(self.seg)
        if self.kind not in ("skirting", "rail"):
            raise ValueError("strip kind must be 'skirting' or 'rail'")
        x0 = 0.0 if self.a0 < 0 else self.a0
        x1 = _length(seg) if self.a1 < 0 else self.a1
        if self.a0 < 0 or self.a1 < 0:
            # a default end stops where the strip would run into a neighbouring wall
            f0, f1 = free_span(seg, x0, x1, self.depth / 2, (self.z0 + self.z1) / 2)
            x0 = f0 if self.a0 < 0 else x0
            x1 = f1 if self.a1 < 0 else x1
        parts = [(s0, s1, 0.0, self.depth, self.z0, self.z1) for s0, s1 in _cut_by_openings(seg, x0, x1, self.z0, self.z1)]
        mat = _material(f"{seg.name}__{self.kind}__{self.item}", self.rgb, self.roughness)
        _box(f"{seg.name}__{self.kind}__{self.item}", seg, parts, mat, self.kind, self.item,
             {"sg_z": [self.z0, self.z1], "sg_along": [x0, x1], "sg_depth": self.depth})
        print(f"SG strip {seg.name} {self.kind} z={self.z0:.3f}-{self.z1:.3f} along={x0:.3f}-{x1:.3f} "
              f"pieces={len(parts)} item={self.item}")
        return {"FINISHED"}


class SG_OT_frame(bpy.types.Operator):
    """Add a door casing around a segment's handoff opening"""
    bl_idname = "sg.frame"
    bl_label = "Door casing"
    bl_options = {"REGISTER", "UNDO"}
    seg: bpy.props.StringProperty()
    width: bpy.props.FloatProperty(default=0.06)
    depth: bpy.props.FloatProperty(default=0.015)
    sides: bpy.props.EnumProperty(items=[("FRONT", "front", ""), ("BACK", "back", ""), ("BOTH", "both", "")], default="FRONT")
    opening: bpy.props.IntProperty(default=0)
    rgb: bpy.props.FloatVectorProperty(size=3, default=(200, 200, 200))
    roughness: bpy.props.FloatProperty(default=0.6, min=0.0, max=1.0)
    item: bpy.props.StringProperty()

    def execute(self, context):
        seg = _segment(self.seg)
        ops = _openings(seg)
        if not ops:
            raise ValueError(f"{seg.name} has no handoff opening; casings go only around handoff openings")
        a0, a1, z0, z1 = ops[self.opening]
        t = float(seg["sg_thickness"])
        w, d = self.width, self.depth
        faces = []
        if self.sides in ("FRONT", "BOTH"):
            faces.append((0.0, d))                    # into the room: +Y
        if self.sides in ("BACK", "BOTH"):
            faces.append((-t - d, -t))                # behind the back face
        parts = []
        for y0, y1 in faces:
            parts += [(a0 - w, a0, y0, y1, z0, z1 + w),          # left jamb
                      (a1, a1 + w, y0, y1, z0, z1 + w),          # right jamb
                      (a0, a1, y0, y1, z1, z1 + w)]              # head
        mat = _material(f"{seg.name}__casing__{self.item}", self.rgb, self.roughness)
        _box(f"{seg.name}__casing__{self.item}", seg, parts, mat, "casing", self.item,
             {"sg_opening": self.opening, "sg_width": w, "sg_depth": d, "sg_sides": self.sides})
        print(f"SG frame {seg.name} opening={self.opening} width={w:.3f} sides={self.sides} item={self.item}")
        return {"FINISHED"}


TILE_PX = 256           # tile image: one tile with its joint on two edges, repeated over the face


def _tile_image(name, size, joint, rgb, joint_rgb):
    """One tile with half a joint on each edge, as an sRGB byte PNG (packed); repeats seamlessly."""
    import numpy as np
    w = TILE_PX
    h = max(8, int(round(TILE_PX * size[1] / size[0])))
    px = np.empty((h, w, 4), np.float32)
    px[...] = [*(max(0.0, min(255.0, float(c))) / 255.0 for c in rgb), 1.0]
    jx = max(1, int(round(w * joint / size[0] / 2)))
    jy = max(1, int(round(h * joint / size[1] / 2)))
    j = [*(max(0.0, min(255.0, float(c))) / 255.0 for c in joint_rgb), 1.0]
    if joint > 0:
        px[:jy], px[-jy:], px[:, :jx], px[:, -jx:] = j, j, j, j
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(name, w, h, alpha=False, float_buffer=False)
    img.colorspace_settings.name = "sRGB"
    img.pixels.foreach_set(px.ravel())
    img.file_format = "PNG"
    img.pack()
    return img


def _uv_layout(ob, face_mat_index, size, angle_deg, offset):
    """Planar UVs (world XY) on the faces of one material: 1 UV unit = one tile, rotated by angle."""
    import math
    me = ob.data
    uv = me.uv_layers.get("st") or me.uv_layers.new(name="st")
    c, s_ = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    mw = ob.matrix_world
    for poly in me.polygons:
        if poly.material_index != face_mat_index:
            continue
        for li in poly.loop_indices:
            p = mw @ me.vertices[me.loops[li].vertex_index].co
            x, y = c * p.x + s_ * p.y, -s_ * p.x + c * p.y
            uv.data[li].uv = (x / size[0], y / size[1])


class SG_OT_tiles(bpy.types.Operator):
    """Lay a procedural tile or plank pattern on one face of a floor (or a wall segment)"""
    bl_idname = "sg.tiles"
    bl_label = "Tiles"
    bl_options = {"REGISTER", "UNDO"}
    seg: bpy.props.StringProperty()
    face: bpy.props.StringProperty(default="top")
    size: bpy.props.FloatVectorProperty(size=2, default=(0.3, 0.3))      # tile / plank length, width in metres
    joint: bpy.props.FloatProperty(default=0.003)                         # joint (grout, gap) width in metres
    rgb: bpy.props.FloatVectorProperty(size=3, default=(200, 200, 200))
    joint_rgb: bpy.props.FloatVectorProperty(size=3, default=(150, 150, 150))
    angle: bpy.props.FloatProperty(default=0.0)                           # degrees, about +Z, in plan
    roughness: bpy.props.FloatProperty(default=0.6, min=0.0, max=1.0)
    item: bpy.props.StringProperty()

    def execute(self, context):
        ob = _body(self.seg)
        name = f"{ob.name}_{self.face}"
        names = [m.name for m in ob.data.materials if m]
        if name not in names:
            raise ValueError(f"{ob.name} has no face {self.face!r}")
        sx, sy = float(self.size[0]), float(self.size[1])
        if not (0.05 <= sx <= 3.0 and 0.05 <= sy <= 3.0):
            raise ValueError(f"tile size {sx:.3f} x {sy:.3f} m is outside 0.05-3 m")
        if not (0.0 <= self.joint <= 0.02):
            raise ValueError(f"joint {self.joint:.4f} m is outside 0-0.02 m")
        mat = _material(name, self.rgb, self.roughness)        # base colour = the tile colour, for flat views
        _plain(mat)
        img = _tile_image(f"{name}_tiles", (sx, sy), self.joint, self.rgb, self.joint_rgb)
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
        tex = nodes.new("ShaderNodeTexImage")
        tex.image, tex.extension, tex.interpolation = img, "REPEAT", "Closest"
        uvmap = nodes.new("ShaderNodeUVMap")
        uvmap.uv_map = "st"
        links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        _uv_layout(ob, names.index(name), (sx, sy), self.angle, 0.0)
        mat["sg_item"] = self.item
        mat["sg_tiles"] = json.dumps({"size_m": [round(sx, 4), round(sy, 4)], "joint_m": round(float(self.joint), 4),
                                      "joint_rgb": [int(round(c)) for c in self.joint_rgb],
                                      "angle_deg": round(float(self.angle), 2)})
        print(f"SG tiles {name} size={sx:.3f}x{sy:.3f} joint={self.joint:.4f} angle={self.angle:.1f} "
              f"rgb={tuple(int(round(c)) for c in self.rgb)} joint_rgb={tuple(int(round(c)) for c in self.joint_rgb)} item={self.item}")
        return {"FINISHED"}


CLASSES = (SG_OT_paint, SG_OT_strip, SG_OT_frame, SG_OT_tiles)


def register():
    for cls in CLASSES:
        if hasattr(bpy.types, cls.__name__):
            try:
                bpy.utils.unregister_class(getattr(bpy.types, cls.__name__))
            except RuntimeError:
                pass
        bpy.utils.register_class(cls)


register()
