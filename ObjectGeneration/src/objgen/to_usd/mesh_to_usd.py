"""Convert a vertex-colored GLB to an upright, bottom-aligned USD asset."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

from ..layout import resolve_mesh_input
from .mesh_ops import GLB_TO_Z_UP, decimate, glb_to_usd_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="Run directory or input GLB")
    parser.add_argument("--output", type=Path, help="Default: <run>/usd/object.usdc")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--height", type=float, help="Set the final height in meters with uniform scaling")
    parser.add_argument("--prim-name", default="Xuebi")
    parser.add_argument("--asset-name", help="Human-readable asset name")
    parser.add_argument("--roughness", type=float, default=0.45,
                        help="PBR roughness in [0, 1] (default 0.45)")
    parser.add_argument("--metallic", type=float, default=0.0,
                        help="PBR metallic in [0, 1] (default 0.0)")
    parser.add_argument("--target-faces", type=int, default=40000,
                        help="Decimate the visual mesh to at most this many faces; "
                             "0 disables decimation (default 40000)")
    args = parser.parse_args()
    try:
        args.mesh, args.output = resolve_mesh_input(args.mesh, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("scale must be finite and positive")
    if args.height is not None and (not np.isfinite(args.height) or args.height <= 0):
        parser.error("height must be finite and positive")
    if args.height is not None and args.scale != 1.0:
        parser.error("choose either --height or --scale")
    if not Sdf.Path.IsValidIdentifier(args.prim_name):
        parser.error("prim-name must be a valid USD identifier")
    if not (0.0 <= args.roughness <= 1.0):
        parser.error("roughness must be in [0, 1]")
    if not (0.0 <= args.metallic <= 1.0):
        parser.error("metallic must be in [0, 1]")
    if args.target_faces < 0:
        parser.error("target-faces must be zero or positive")
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    source_faces = len(mesh.faces)
    mesh = decimate(mesh, args.target_faces)
    rotation = GLB_TO_Z_UP
    scale = args.scale if args.height is None else args.height / np.ptp(mesh.vertices[:, 1])
    points, offset = glb_to_usd_points(mesh.vertices, scale)
    normals = np.asarray(mesh.vertex_normals @ rotation.T, dtype=np.float32)
    colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    colors = np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(args.output.resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = Sdf.Path("/" + args.prim_name)
    visual_path = root_path.AppendPath("Looks/Bottle" if args.prim_name == "Xuebi" else "Looks/Visual")
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetAssetInfoByKey("name", args.asset_name or ("Xuebi bottle" if args.prim_name == "Xuebi" else args.prim_name))
    geom = UsdGeom.Mesh.Define(stage, root_path.AppendChild("Mesh"))
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).ravel()))
    geom.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geom.CreateDoubleSidedAttr(True)
    geom.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(axis=0), points.max(axis=0)], dtype=np.float32)))
    geom.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(colors))
    geom.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set(Vt.FloatArray([1.0]))
    material = UsdShade.Material.Define(stage, visual_path)
    shader = UsdShade.Shader.Define(stage, visual_path.AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(args.roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(args.metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    reader = UsdShade.Shader.Define(stage, visual_path.AppendChild("Color"))
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
    color_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(color_output)
    surface = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface)
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    loaded = Usd.Stage.Open(str(args.output.resolve()))
    check = UsdGeom.Mesh(loaded.GetPrimAtPath(root_path.AppendChild("Mesh")))
    valid, reason = UsdGeom.Mesh.ValidateTopology(check.GetFaceVertexIndicesAttr().Get(), check.GetFaceVertexCountsAttr().Get(), len(check.GetPointsAttr().Get()))
    assert valid, reason
    assert UsdShade.MaterialBindingAPI(check.GetPrim()).ComputeBoundMaterial()[0]
    report = {"source": str(args.mesh.resolve()), "usd": str(args.output.resolve()), "default_prim": str(root_path), "up_axis": "Z", "meters_per_unit": 1.0, "scale": float(scale), "height_m": float(points[:, 2].max()), "dimensions_m": np.ptp(points, axis=0).tolist(), "vertices": len(points), "faces": len(mesh.faces), "source_faces": source_faces, "target_faces": args.target_faces, "decimated": len(mesh.faces) < source_faces, "vertex_colors": len(colors), "topology_valid": valid, "roughness": args.roughness, "metallic": args.metallic, "format": "usdc" if args.output.suffix == ".usdc" else "usda", "physics_configured": False, "glb_to_usd": {"rotation": rotation.tolist(), "scale": float(scale), "offset_m": offset.tolist(), "formula": "p_usd = rotation @ v_glb * scale - offset_m"}}
    args.output.with_suffix(".conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
