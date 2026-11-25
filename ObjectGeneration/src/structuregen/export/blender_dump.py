"""Runs inside Blender: dump the finished appearance model for the result page.

    blender -b <agent run>/appearance.blend --python src/structuregen/export/blender_dump.py -- <out.json>

Every mesh as world-space triangles with the sRGB colour the agent gave its face
(material property ``sg_rgb``), and every object's ``sg_*`` properties, so the page
reads the model itself rather than the agent's own account of it.
"""
import json
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]


def plain(value):
    if hasattr(value, "to_list"):
        return value.to_list()
    if hasattr(value, "__len__") and not isinstance(value, str):
        return [plain(v) for v in value]
    return value


def roughness(m):
    """The Principled BSDF value sg.paint set; the viewport field stays at Blender's default."""
    if m is None or not m.use_nodes:
        return None
    bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    return round(bsdf.inputs["Roughness"].default_value, 3) if bsdf else None


objects = []
for ob in sorted(bpy.data.objects, key=lambda o: o.name):
    if ob.type != "MESH":
        continue
    mesh = ob.data
    mesh.calc_loop_triangles()
    mw = ob.matrix_world
    slots = []
    for slot in ob.material_slots:
        m = slot.material
        slots.append({"name": m.name if m else None,
                      "rgb": list(m.get("sg_rgb", [])) if m else [],
                      "item": m.get("sg_item") if m else None,
                      "roughness": roughness(m),
                      "tiles": json.loads(m["sg_tiles"]) if m and "sg_tiles" in m else None})
    tris = []
    for t in mesh.loop_triangles:
        pts = [list(mw @ mesh.vertices[i].co) for i in t.vertices]
        tris.append({"p": [[round(c, 5) for c in p] for p in pts], "m": t.material_index})
    props = {k: plain(ob[k]) for k in ob.keys() if k.startswith("sg_") and k != "sg_parts"}
    objects.append({"name": ob.name, "parent": ob.parent.name if ob.parent else None,
                    "slots": slots, "props": props, "triangles": tris})

with open(out, "w") as f:
    json.dump({"objects": objects}, f)
print("DUMPED", len(objects), "objects ->", out)
