"""Runs inside Blender: the model half of a compare image, as separate passes.

    blender -b <scene.blend> --python src/structuregen/agent/blender/blender_compare.py -- <out dir> <scale> <camera> [<camera> ...]

Writes per camera, at `scale` x the photo size:

    <cam>_flat.png  Workbench, FLAT light, texture colour: every pixel is exactly its material's base
                    colour, or its tile pattern where sg.tiles laid one
    <cam>_ao.png    Cycles, ambient occlusion only (white where open, darker in corners)
    <cam>_geo.png   16-bit: surface normal in camera space (RGB)
    <cam>_dist.png  16-bit: R = distance to the camera / 50 m, G = 1 on trims (non-body objects)

Nothing is saved back: materials are only read (the flat pass copies each Principled base colour
into the viewport colour of this in-memory scene), the other passes use a view-layer override.
`compare.py` composes them into flat x AO with thin outlines.
"""
import sys
from pathlib import Path

import bpy


def render_passes(out, scale, cams):
    """Render flat / ao / geo / dist for every camera into `out`, then restore the scene.

    Touches render settings, the view-layer override, viewport colours and pass indices, and
    puts every one of them back, so a long-lived scene (the harness worker) is left as it was.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scene, layer = bpy.context.scene, bpy.context.view_layer
    r = scene.render
    shading = scene.display.shading
    saved = {
        "engine": r.engine, "res": r.resolution_percentage, "film": r.film_transparent,
        "dither": r.dither_intensity, "fmt": r.image_settings.file_format,
        "mode": r.image_settings.color_mode, "depth": r.image_settings.color_depth,
        "vt": scene.view_settings.view_transform, "look": scene.view_settings.look,
        "aa": scene.display.render_aa, "camera": scene.camera, "override": layer.material_override,
        "shading": {k: getattr(shading, k) for k in ("light", "color_type", "show_shadows", "show_cavity",
                                                  "show_object_outline", "show_specular_highlight")},
        "path": r.filepath,
        "diffuse": {m.name: tuple(m.diffuse_color) for m in bpy.data.materials},
        "pass_index": {o.name: o.pass_index for o in bpy.data.objects if o.type == "MESH"},
    }
    made = []
    try:
        r.resolution_percentage = int(round(scale * 100))
        r.film_transparent = True
        r.dither_intensity = 0
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGBA"
        scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"
        for ob in bpy.data.objects:
            if ob.type == "MESH":
                ob.pass_index = 0 if ob.parent is None else 1      # trims are children of a segment
        for mat in bpy.data.materials:
            bsdf = next((n for n in (mat.node_tree.nodes if mat.node_tree else []) if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is not None:
                mat.diffuse_color = tuple(bsdf.inputs["Base Color"].default_value)
        overrides = {k: _override(k) for k in ("ao", "geo", "dist")}
        made = list(overrides.values())
        raw = "Raw" if "Raw" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items] else "Standard"
        for name in cams:
            scene.camera = bpy.data.objects[name]
            for kind in ("flat", "ao", "geo", "dist"):
                if kind == "flat":
                    r.engine = "BLENDER_WORKBENCH"
                    shading.light, shading.color_type = "FLAT", "TEXTURE"      # tiles show; plain = base colour
                    shading.show_shadows = shading.show_cavity = False
                    shading.show_object_outline = shading.show_specular_highlight = False
                    scene.display.render_aa = "8"
                    layer.material_override = None
                    r.image_settings.color_depth = "8"
                    scene.view_settings.view_transform = "Standard"
                else:
                    r.engine = "CYCLES"
                    c = scene.cycles
                    c.device, c.use_denoising, c.use_adaptive_sampling, c.max_bounces = "CPU", False, False, 0
                    c.samples = 32 if kind == "ao" else 1
                    c.filter_width = 1.5 if kind == "ao" else 0.01      # edge passes stay aliased: no mixed values
                    layer.material_override = overrides[kind]
                    r.image_settings.color_depth = "8" if kind == "ao" else "16"
                    scene.view_settings.view_transform = "Standard" if kind == "ao" else raw
                r.filepath = str(out / f"{name}_{kind}.png")
                bpy.ops.render.render(write_still=True)
    finally:
        r.engine, r.resolution_percentage, r.film_transparent = saved["engine"], saved["res"], saved["film"]
        r.dither_intensity, r.filepath = saved["dither"], saved["path"]
        r.image_settings.file_format, r.image_settings.color_mode = saved["fmt"], saved["mode"]
        r.image_settings.color_depth = saved["depth"]
        scene.view_settings.view_transform, scene.view_settings.look = saved["vt"], saved["look"]
        scene.display.render_aa = saved["aa"]
        scene.camera, layer.material_override = saved["camera"], saved["override"]
        for k, v in saved["shading"].items():
            setattr(shading, k, v)
        for name, colour in saved["diffuse"].items():
            m = bpy.data.materials.get(name)
            if m is not None:
                m.diffuse_color = colour
        for name, index in saved["pass_index"].items():
            o = bpy.data.objects.get(name)
            if o is not None:
                o.pass_index = index
        for m in made:
            bpy.data.materials.remove(m)


def _override(kind):
    mat = bpy.data.materials.new(f"_sg_compare_{kind}")
    mat.use_nodes = True
    t = mat.node_tree
    t.nodes.clear()
    out = t.nodes.new("ShaderNodeOutputMaterial")
    em = t.nodes.new("ShaderNodeEmission")
    t.links.new(em.outputs[0], out.inputs["Surface"])
    if kind == "ao":
        ao = t.nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples = 16
        ao.inputs["Distance"].default_value = 0.35
        t.links.new(ao.outputs["AO"], em.inputs["Color"])
    elif kind == "geo":
        geo = t.nodes.new("ShaderNodeNewGeometry")
        vt = t.nodes.new("ShaderNodeVectorTransform")
        vt.vector_type, vt.convert_from, vt.convert_to = "NORMAL", "WORLD", "CAMERA"
        mul = t.nodes.new("ShaderNodeVectorMath")
        mul.operation = "MULTIPLY_ADD"
        mul.inputs[1].default_value = (0.5, 0.5, 0.5)
        mul.inputs[2].default_value = (0.5, 0.5, 0.5)
        t.links.new(geo.outputs["Normal"], vt.inputs[0])
        t.links.new(vt.outputs[0], mul.inputs[0])
        t.links.new(mul.outputs[0], em.inputs["Color"])
    else:
        cam = t.nodes.new("ShaderNodeCameraData")
        info = t.nodes.new("ShaderNodeObjectInfo")
        div = t.nodes.new("ShaderNodeMath")
        div.operation = "DIVIDE"
        div.inputs[1].default_value = 50.0
        comb = t.nodes.new("ShaderNodeCombineXYZ")
        t.links.new(cam.outputs["View Distance"], div.inputs[0])
        t.links.new(div.outputs[0], comb.inputs[0])
        t.links.new(info.outputs["Object Index"], comb.inputs[1])
        t.links.new(comb.outputs[0], em.inputs["Color"])
    return mat


if __name__ == "__main__" and "--" in sys.argv:
    _args = sys.argv[sys.argv.index("--") + 1:]
    render_passes(Path(_args[0]), float(_args[1]), _args[2:])
    print("SG_COMPARE_PASSES", len(_args) - 2, flush=True)
