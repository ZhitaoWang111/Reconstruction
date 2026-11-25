"""Author a simulation-ready articulated USD from a DualCraft product.

The product's URDF gives the structure; ``dualcraft_product`` rebuilds the physics.
This writes one USD per object, laid out for PhysX:

    /<Prim>                     Xform, defaultPrim, origin = asset origin (XY box centre,
      │                         lowest point on z = 0), PhysicsArticulationRootAPI
      ├── Looks/                visual materials as DualCraft authored them, plus the
      │                         physics material bound to every collider
      ├── <body>                one Xform per rigid body (root + every movable link,
      │   │                     each with the fixed-joint links merged into it), all
      │   │                     siblings: PhysX does not nest rigid bodies.
      │   │                     RigidBodyAPI + MassAPI (mass, centre of mass, inertia)
      │   ├── visual_*          meshes of every merged link, materials bound
      │   └── collision/hull_*  one convex mesh per piece, invisible
      └── joints/               PhysicsRevoluteJoint / PhysicsPrismaticJoint per movable
                                joint, and ``fix_base`` (static root) to the world

The visual geometry is read with DualCraft's own OBJ / MTL readers and material
authoring, so the look matches its object.usdc; only the hierarchy is flattened.

Usage::

    PYTHONPATH=src .venv-sam3d/bin/python -m objgen.dualcraft.configure_articulated_physics \\
        <product dir> --output physics.usdc --mass 52 --body-type static \\
        --blueprint-run <dir with inputs/prompt.json>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from ..config import DUALCRAFT_REPO
sys.path.insert(0, str(DUALCRAFT_REPO / "src"))

from . import dualcraft_product as dp  # noqa: E402
from sim_export import urdf2usd  # noqa: E402

# Same physics material as configure_usd_physics.py, so both generators collide alike.
STATIC_FRICTION, DYNAMIC_FRICTION, RESTITUTION = 0.6, 0.5, 0.0


def gf_matrix(matrix: np.ndarray) -> Gf.Matrix4d:
    """Column-vector 4x4 -> Gf (row vectors): transpose."""
    return Gf.Matrix4d(*[float(v) for v in np.asarray(matrix, dtype=np.float64).T.reshape(-1)])


def gf_quat(rotation: np.ndarray) -> Gf.Quatf:
    quat = Gf.Matrix3d(*[float(v) for v in np.asarray(rotation).T.reshape(-1)]).ExtractRotation().GetQuat()
    return Gf.Quatf(float(quat.GetReal()), Gf.Vec3f(*[float(v) for v in quat.GetImaginary()]))


def identifier(name: str) -> str:
    return urdf2usd._identifier(name, "part")


def local_texture(material, out_dir: Path):
    """Copy a material's texture under ``out_dir/textures`` and point at the copy, so
    the USD keeps working when its directory is moved without the product."""
    if material.texture is None:
        return material
    target = out_dir / "textures" / f"{material.texture.parent.name}_{material.texture.name}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(material.texture, target)
    return dataclasses.replace(material, texture=target)


def author_visuals(stage: Usd.Stage, body_path: Sdf.Path, product: dp.Product, body: str,
                   members: list[str], looks: Sdf.Path, out_dir: Path, cache: dict) -> int:
    """Meshes of every link merged into ``body``, in the body frame."""
    mtl_cache: dict[Path, dict] = {}
    meshes = 0
    for link in members:
        offset = dp.link_in_body(product, link, body)
        for order, (path, transform) in enumerate(product.links[link].visuals):
            obj = urdf2usd.read_obj(path)
            materials = {}
            for reference in obj.mtllib:
                mtl = (path.parent / reference).resolve()
                if mtl.is_file():
                    mtl_cache.setdefault(mtl, urdf2usd.read_mtl(mtl))
                    materials.update(mtl_cache[mtl])
            placement = gf_matrix(offset @ transform)
            for index, group in enumerate(obj.groups):
                mesh = urdf2usd.author_mesh_group(
                    stage, body_path, identifier(f"visual_{link}_{order}_{index}"), obj, group,
                    (1.0, 1.0, 1.0), placement)
                chosen = materials.get(group.material or "") or urdf2usd.Material(name=f"{link}_default")
                chosen = local_texture(chosen, out_dir)
                material_path = urdf2usd.author_material(stage, looks, chosen, out_dir, cache)
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                    UsdShade.Material.Get(stage, material_path))
                # No CollisionAPI at all: PhysX parses any prim carrying it, even with
                # collisionEnabled false, and turns triangle meshes on a moving body
                # into extra convex hulls.  Only collision/hull_* collide.
                meshes += 1
    return meshes


def author_hulls(stage: Usd.Stage, body_path: Sdf.Path, hulls: list, material: UsdShade.Material) -> int:
    scope = UsdGeom.Scope.Define(stage, body_path.AppendChild("collision"))
    UsdGeom.Imageable(scope.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    for index, hull in enumerate(hulls):
        mesh = UsdGeom.Mesh.Define(stage, scope.GetPath().AppendChild(f"hull_{index:03d}"))
        points = np.asarray(hull.vertices, dtype=np.float32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(hull.faces), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(hull.faces, dtype=np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], dtype=np.float32)))
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        # Each piece is convex already, so the hull approximation cooks it exactly.
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    return len(hulls)


def joint_frame(axis: np.ndarray) -> tuple[str, np.ndarray]:
    """USD joints move along X, Y or Z of their local frame.  Pick the frame axis the
    URDF axis lies on (with its sign folded into a rotation), so limits keep their
    meaning; an oblique axis gets a frame whose X is the axis."""
    for token, index in (("X", 0), ("Y", 1), ("Z", 2)):
        if abs(abs(axis[index]) - 1.0) < 1e-6:
            if axis[index] > 0:
                return token, np.eye(3)
            # Rotate 180 degrees about an axis perpendicular to this one: the joint
            # axis flips sign while staying on the same frame axis.
            other = np.eye(3)[(index + 1) % 3]
            return token, 2 * np.outer(other, other) - np.eye(3)
    x = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    return "X", np.column_stack([x, y, np.cross(x, y)])


def author_joint(stage: Usd.Stage, path: Sdf.Path, joint: dp.Joint, product: dp.Product,
                 groups: dict[str, list[str]], body_paths: dict[str, Sdf.Path]):
    group_of = {link: body for body, members in groups.items() for link in members}
    parent_body, child_body = group_of[joint.parent], group_of[joint.child]
    token, local = joint_frame(joint.axis)
    # Joint frame expressed in the child body frame (the child link frame is the
    # child body frame), and in the parent body frame at rest.
    in_child = np.eye(4)
    in_child[:3, :3] = local
    in_parent = dp.link_in_body(product, joint.parent, parent_body) @ joint.origin @ in_child
    if joint.type in ("revolute", "continuous"):
        usd_joint = UsdPhysics.RevoluteJoint.Define(stage, path)
        if joint.type == "revolute" and joint.lower is not None and joint.upper is not None:
            usd_joint.CreateLowerLimitAttr(math.degrees(joint.lower))
            usd_joint.CreateUpperLimitAttr(math.degrees(joint.upper))
    else:
        usd_joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        if joint.lower is not None and joint.upper is not None:
            usd_joint.CreateLowerLimitAttr(joint.lower)
            usd_joint.CreateUpperLimitAttr(joint.upper)
    usd_joint.CreateAxisAttr(token)
    usd_joint.CreateBody0Rel().SetTargets([body_paths[parent_body]])
    usd_joint.CreateBody1Rel().SetTargets([body_paths[child_body]])
    usd_joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in in_parent[:3, 3]]))
    usd_joint.CreateLocalRot0Attr(gf_quat(in_parent[:3, :3]))
    usd_joint.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in in_child[:3, 3]]))
    usd_joint.CreateLocalRot1Attr(gf_quat(in_child[:3, :3]))
    prim = usd_joint.GetPrim()
    prim.SetCustomDataByKey("urdf_joint", {"name": joint.name, "type": joint.type,
                                           "axis": [float(v) for v in joint.axis],
                                           "parent_link": joint.parent, "child_link": joint.child})
    return usd_joint


def write_stage(output: Path, product: dp.Product, bodies: dict, groups: dict, offset: np.ndarray,
                prim_name: str, body_type: str, *, physics: bool) -> dict:
    """One USD with a flat Xform per body.  ``physics=False`` writes the same layout
    with visuals only (the visual.usdc slot); ``True`` adds bodies, colliders, joints."""
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = Sdf.Path(f"/{prim_name}")
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    looks = root_path.AppendChild("Looks")
    UsdGeom.Scope.Define(stage, looks)
    if physics:
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        physics_material = UsdShade.Material.Define(stage, looks.AppendChild("Physics"))
        material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
        material_api.CreateStaticFrictionAttr(STATIC_FRICTION)
        material_api.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
        material_api.CreateRestitutionAttr(RESTITUTION)

    # Every body sits at its rest pose in the asset frame (DualCraft frame minus offset).
    asset_from_dualcraft = dp.translation(-offset)
    body_paths, records, visual_cache = {}, {}, {}
    for name, body in bodies.items():
        path = root_path.AppendChild(identifier(name))
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTransformOp().Set(gf_matrix(asset_from_dualcraft @ dp.body_in_root(product, name)))
        prim = xform.GetPrim()
        prim.SetCustomDataByKey("dualcraft_links", ",".join(body.links))
        body_paths[name] = path
        meshes = author_visuals(stage, path, product, name, body.links, looks, output.parent, visual_cache)
        records[name] = {"prim": str(path), "links": body.links, "visual_meshes": meshes}
        if not physics:
            continue
        records[name]["collision_hulls"] = author_hulls(stage, path, body.hulls, physics_material)
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        moments, axes_matrix = dp.principal(body.inertia)
        mass_api.CreateMassAttr(float(body.mass))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in body.centre_of_mass]))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in moments]))
        mass_api.CreatePrincipalAxesAttr(gf_quat(axes_matrix))
        records[name].update({
            "mass_kg": round(body.mass, 4),
            "material_volumes_m3": {k: round(v, 6) for k, v in body.material_volumes.items()},
            "center_of_mass_m": np.round(body.centre_of_mass, 5).tolist(),
            "principal_inertia_kg_m2": np.round(moments, 6).tolist(),
            "pieces": body.pieces,
        })

    if physics:
        joints_scope = root_path.AppendChild("joints")
        UsdGeom.Scope.Define(stage, joints_scope)
        for joint in product.joints:
            if joint.type != "fixed":
                author_joint(stage, joints_scope.AppendChild(identifier(joint.name)), joint, product,
                             groups, body_paths)
        if body_type == "static":
            # Fix the root body to the world.  body0 unset means the world frame; the
            # joint frames coincide with the root body's rest pose in the asset frame.
            fix = UsdPhysics.FixedJoint.Define(stage, joints_scope.AppendChild("fix_base"))
            fix.CreateBody1Rel().SetTargets([body_paths[product.root]])
            rest = asset_from_dualcraft @ dp.body_in_root(product, product.root)
            fix.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in rest[:3, 3]]))
            fix.CreateLocalRot0Attr(gf_quat(rest[:3, :3]))
            fix.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            fix.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    root.GetPrim().SetCustomDataByKey("dualcraft", {
        "robot": product.robot, "root_body": product.root,
        "asset_offset_m": [float(v) for v in offset],
        "note": "p_dualcraft = p_asset + asset_offset_m; DualCraft frame is +X front, +Y left, +Z up"})
    stage.GetRootLayer().Export(str(output))
    return records


def build(product_dir: Path, output: Path, mass_kg: float, body_type: str, blueprint_run: Path,
          prim_name: str, trajectory: Path | None = None, visual_output: Path | None = None) -> dict:
    if body_type not in ("static", "dynamic"):
        raise ValueError("articulated assets take body_type static (root fixed) or dynamic")
    product = dp.read_product(product_dir)
    materials = dp.link_materials(product, dp.read_blueprint(blueprint_run))
    groups = dp.rigid_groups(product)
    bodies = dp.build_bodies(product, materials)
    mass_summary = dp.distribute_mass(bodies, mass_kg)
    bounds = dp.rest_bounds(product)
    offset = dp.asset_offset(bounds)
    checks = dp.interference(product, bodies)
    axes = dp.axis_provenance(trajectory)
    output = Path(output)
    records = write_stage(output, product, bodies, groups, offset, prim_name, body_type, physics=True)
    if visual_output is not None:
        visual_output = Path(visual_output)
        visual = write_stage(visual_output, product, bodies, groups, offset, prim_name, body_type, physics=False)
        visual_output.with_suffix(".conversion.json").write_text(json.dumps({
            "source": str(product_dir), "usd": str(visual_output), "generator": "dualcraft",
            "dimensions_m": np.round(bounds[1] - bounds[0], 4).tolist(),
            "asset_offset_m": np.round(offset, 6).tolist(),
            "frame": "asset: DualCraft object frame shifted by -asset_offset_m, so the XY bounding "
                     "box is centred on the origin and the lowest point is at z=0",
            "visual_meshes": sum(r["visual_meshes"] for r in visual.values()),
            "layout": "one Xform per rigid body, siblings, at the rest pose (doors closed)",
        }, indent=2) + "\n")

    joint_records = [{
        "name": joint.name, "type": joint.type, "parent": joint.parent, "child": joint.child,
        "axis": [round(float(v), 6) for v in joint.axis],
        "lower": joint.lower, "upper": joint.upper,
        "urdf_effort": joint.effort, "urdf_velocity": joint.velocity,
        "axis_source": axes.get(joint.name, {}).get("source") or ("unknown" if not axes else "default"),
        "axis_changed_by_agent": axes.get(joint.name, {}).get("changed"),
    } for joint in product.joints if joint.type != "fixed"]
    implied = mass_summary["implied_density_kg_m3"]
    density_flags = [f"{m} implied {d:.0f} kg/m3 outside {lo:.0f}-{hi:.0f}"
                     for m, d in implied.items()
                     for lo, hi in [dp.DENSITY_RANGE_KG_M3[m]] if not lo <= d <= hi]
    report = {
        "source": str(product_dir), "asset": str(output), "physics_configured": True,
        "articulated": True, "body_type": body_type, "fixed_base": body_type == "static",
        "mass_kg": round(sum(b.mass for b in bodies.values()), 4), "requested_mass_kg": mass_kg,
        "mass_distribution_assumption": "each body: convex-hull volume of its pieces x material "
                                        "density (wood 600, metal 7800, ...), one factor scaling "
                                        "all densities to the total; minimum 0.02 kg per body",
        **mass_summary, "density_flags": density_flags,
        "collision_approximation": "convexHullPerPiece",
        "collision_hulls": sum(r["collision_hulls"] for r in records.values()),
        "visual_collides": False,
        "static_friction": STATIC_FRICTION, "dynamic_friction": DYNAMIC_FRICTION, "restitution": RESTITUTION,
        "asset_offset_m": np.round(offset, 6).tolist(),
        "dimensions_m": np.round(bounds[1] - bounds[0], 4).tolist(),
        "link_materials": materials, "bodies": records, "joints": joint_records,
        "interference": checks,
        "joint_dynamics": "not authored (no damping, no friction, no drive); URDF effort / velocity kept in the report only",
    }
    output.with_suffix(".physics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("product", type=Path, help="DualCraft product directory (model.urdf + assets/)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mass", type=float, required=True, help="total mass in kg")
    parser.add_argument("--body-type", choices=["static", "dynamic"], required=True,
                        help="static fixes the root body to the world; doors still move")
    parser.add_argument("--blueprint-run", type=Path, required=True,
                        help="directory holding inputs/prompt.json (part finishes)")
    parser.add_argument("--trajectory", type=Path, help="DualCraft trajectory, for joint-axis provenance")
    parser.add_argument("--visual-output", type=Path,
                        help="also write the same layout without physics (the visual.usdc slot)")
    parser.add_argument("--prim-name", default="Object")
    args = parser.parse_args()
    if not math.isfinite(args.mass) or args.mass <= 0:
        parser.error("mass must be finite and positive")
    report = build(args.product, args.output, args.mass, args.body_type, args.blueprint_run,
                   args.prim_name, args.trajectory, args.visual_output)
    print(json.dumps({k: report[k] for k in ("asset", "body_type", "mass_kg", "implied_density_kg_m3",
                                              "density_flags", "collision_hulls", "interference")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
