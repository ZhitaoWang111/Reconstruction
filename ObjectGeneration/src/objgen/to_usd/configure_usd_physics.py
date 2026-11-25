"""Add a dynamic rigid body, mesh collision and mass properties to a USD asset."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from ..layout import resolve_physics_input
from .mesh_ops import convex_hull_from_mesh, decimate


def build_collision_proxy(visual: trimesh.Trimesh, approximation: str,
                          collision_faces: int) -> trimesh.Trimesh:
    """Return the mesh PhysX should cook, independent of the visual mesh.

    ``convexHull`` samples the surface and takes the hull directly, which
    collapses a 1.4M-face mesh to a few hundred faces.  ``convexDecomposition``
    needs real concave geometry to decompose, so it gets a decimated copy of
    the visual mesh instead of the hull.
    """
    if approximation == "convexHull":
        return convex_hull_from_mesh(visual)
    return decimate(visual, collision_faces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path, help="Run directory or visual USD asset")
    parser.add_argument("--output", type=Path, help="Default: <run>/physics/<asset-name>; legacy input uses <name>_physics.usd[a]")
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--collision-approximation", choices=["convexHull", "convexDecomposition"], default="convexHull")
    parser.add_argument("--max-convex-hulls", type=int, default=128)
    parser.add_argument("--voxel-resolution", type=int, default=500000)
    parser.add_argument("--solver-velocity-iterations", type=int,
                        help="Optional PhysX rigid-body velocity solver iterations (0-255); e.g. 8 reduces contact velocity residuals")
    parser.add_argument("--solver-position-iterations", type=int,
                        help="Optional PhysX rigid-body position solver iterations (1-255); e.g. 64 improves contact convergence")
    parser.add_argument("--body-type", choices=["dynamic", "kinematic", "static"],
                        default="dynamic",
                        help="dynamic = simulated rigid body; kinematic = transform-driven "
                             "body that still pushes others; static = collision-only, "
                             "no RigidBodyAPI authored (default dynamic)")
    parser.add_argument("--kinematic", action="store_true",
                        help="Deprecated alias for --body-type kinematic")
    parser.add_argument("--collision-faces", type=int, default=5000,
                        help="Face budget for the collision proxy under convexDecomposition; "
                             "0 keeps the visual mesh as-is (default 5000)")
    args = parser.parse_args()
    if args.kinematic:
        if args.body_type not in ("kinematic", "dynamic"):
            parser.error("--kinematic conflicts with --body-type static")
        args.body_type = "kinematic"
    try:
        args.usd, args.output = resolve_physics_input(args.usd, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    if not np.isfinite(args.mass) or args.mass <= 0:
        parser.error("mass must be finite and positive")
    if args.max_convex_hulls < 1 or args.voxel_resolution < 10000:
        parser.error("convex hull count must be positive and voxel resolution at least 10000")
    if args.solver_velocity_iterations is not None and not 0 <= args.solver_velocity_iterations <= 255:
        parser.error("solver velocity iterations must be between 0 and 255")
    if args.solver_position_iterations is not None and not 1 <= args.solver_position_iterations <= 255:
        parser.error("solver position iterations must be between 1 and 255")
    stage = Usd.Stage.Open(str(args.usd.resolve()))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = stage.GetDefaultPrim()
    geom = UsdGeom.Mesh(stage.GetPrimAtPath(root.GetPath().AppendChild("Mesh")))
    vertices = np.asarray(geom.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(geom.GetFaceVertexCountsAttr().Get())
    assert np.all(counts == 3)
    faces = np.asarray(geom.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    body_type = args.body_type
    if body_type == "dynamic":
        assert mesh.is_watertight and mesh.volume > 0
        mesh.density = args.mass / mesh.volume
        inertia = mesh.moment_inertia
        moments, axes = np.linalg.eigh(inertia)
        assert np.isfinite(inertia).all() and np.all(moments > 0)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        # Gf matrices act on row vectors; eigenvectors above are column vectors.
        rotation = Gf.Matrix3d(*axes.T.ravel().tolist())
        orientation = Gf.Quatf(rotation.ExtractRotation().GetQuat())
        reconstructed_axes = np.asarray(Gf.Matrix3d(1).SetRotate(Gf.Quatd(orientation))).T
        # USD principalAxes uses a float quaternion; allow its rounding error at the inertia's scale.
        assert np.allclose(reconstructed_axes @ np.diag(moments) @ reconstructed_axes.T, inertia, atol=max(1e-9, float(moments.max()) * 1e-7), rtol=1e-6)
        center_of_mass = mesh.center_mass
    else:
        # kinematic and static skip inertia.
        moments, orientation, center_of_mass = None, None, None

    if body_type in ("dynamic", "kinematic"):
        body = UsdPhysics.RigidBodyAPI.Apply(root)
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(body_type == "kinematic")
        if args.solver_velocity_iterations is not None:
            root.AddAppliedSchema("PhysxRigidBodyAPI")
            root.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int,
                                 custom=False).Set(args.solver_velocity_iterations)
        if args.solver_position_iterations is not None:
            root.AddAppliedSchema("PhysxRigidBodyAPI")
            root.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int,
                                 custom=False).Set(args.solver_position_iterations)
        mass_api = UsdPhysics.MassAPI.Apply(root)
        mass_api.CreateMassAttr(args.mass)
        if center_of_mass is not None:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments))
            mass_api.CreatePrincipalAxesAttr(orientation)
    # static: no RigidBodyAPI, no MassAPI — pure collision geometry
    # Visual mesh: keep it as pure rendering geometry.  Collision lives on a
    # separate, much cheaper proxy prim so PhysX never cooks 100k+ triangles.
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(False)

    collision_mesh = build_collision_proxy(mesh, args.collision_approximation,
                                           args.collision_faces)
    collision_path = root.GetPath().AppendChild("CollisionMesh")
    collision = UsdGeom.Mesh.Define(stage, collision_path)
    collision_points = np.asarray(collision_mesh.vertices, dtype=np.float32)
    collision.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(collision_points))
    collision.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(collision_mesh.faces), 3, dtype=np.int32)))
    collision.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(collision_mesh.faces, dtype=np.int32).ravel()))
    collision.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    collision.CreateDoubleSidedAttr(False)
    collision.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([collision_points.min(axis=0), collision_points.max(axis=0)], dtype=np.float32)))
    UsdGeom.Imageable(collision.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr(args.collision_approximation)
    decomposition = None
    if args.collision_approximation == "convexDecomposition":
        # Author the documented PhysX API without importing Isaac Sim into the SAM3D runtime.
        # Isaac Sim resolves this schema and cooks the compound hulls when simulation starts.
        collision.GetPrim().AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
        decomposition = {"maxConvexHulls": args.max_convex_hulls, "voxelResolution": args.voxel_resolution, "errorPercentage": 1.0, "minThickness": 0.001, "shrinkWrap": True}
        for key, value in decomposition.items():
            attr_type = Sdf.ValueTypeNames.Bool if isinstance(value, bool) else Sdf.ValueTypeNames.Int if isinstance(value, int) else Sdf.ValueTypeNames.Float
            collision.GetPrim().CreateAttribute("physxConvexDecompositionCollision:" + key, attr_type, custom=False).Set(value)
    material = UsdShade.Material.Define(stage, root.GetPath().AppendPath("Looks/Physics"))
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.6)
    physics_material.CreateDynamicFrictionAttr(0.5)
    physics_material.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(material, materialPurpose="physics")
    # Export the edited in-memory stage; never save over the stage-4 source.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert stage.Export(str(args.output))
    loaded = Usd.Stage.Open(str(args.output))
    check = loaded.GetDefaultPrim()
    check_mesh = loaded.GetPrimAtPath(geom.GetPath())
    check_collision = loaded.GetPrimAtPath(collision.GetPath())
    if body_type in ("dynamic", "kinematic"):
        assert check.HasAPI(UsdPhysics.RigidBodyAPI)
        assert np.isclose(UsdPhysics.MassAPI(check).GetMassAttr().Get(), args.mass, rtol=1e-6, atol=0)
    else:
        assert not check.HasAPI(UsdPhysics.RigidBodyAPI), "static body must not have RigidBodyAPI"
    # Visual mesh must not collide; the proxy must, and must carry the physics material.
    assert not UsdPhysics.CollisionAPI(check_mesh).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.CollisionAPI(check_collision).GetCollisionEnabledAttr().Get()
    assert UsdShade.MaterialBindingAPI(check_collision).ComputeBoundMaterial("physics")[0].GetPath() == material.GetPath()
    # The proxy is bound for physics only; the visual mesh keeps its own material.
    assert UsdShade.MaterialBindingAPI(check_mesh).ComputeBoundMaterial()[0].GetPath() != material.GetPath()
    report = {
        "source": str(args.usd), "asset": str(args.output), "physics_configured": True,
        "body_type": body_type,
        "mass_kg": args.mass if body_type in ("dynamic", "kinematic") else None,
        "rigid_body_enabled": body_type in ("dynamic", "kinematic"),
        "kinematic": body_type == "kinematic",
        "collision_enabled": True, "collision_approximation": args.collision_approximation,
        "collision_prim": str(collision.GetPath()),
        "visual_faces": int(len(mesh.faces)),
        "collision_faces": int(len(collision_mesh.faces)),
        "visual_collides": False,
        "static_friction": 0.6, "dynamic_friction": 0.5, "restitution": 0.0,
        "center_of_mass_m": center_of_mass.tolist() if center_of_mass is not None else None,
        "diagonal_inertia_kg_m2": moments.tolist() if moments is not None else None,
        "principal_axes_wxyz": ([float(orientation.GetReal()), *map(float, orientation.GetImaginary())]
                               if orientation is not None else None),
        "mass_distribution_assumption": ({
            "dynamic": "uniform density inside the closed reconstructed mesh",
            "kinematic": "none (kinematic)",
            "static": "none (static)",
        }[body_type]),
        "height_m": float(mesh.extents[2]), "geometry_and_visual_material_preserved": True,
        "convex_decomposition_settings": decomposition,
        "solver_velocity_iterations": args.solver_velocity_iterations,
        "solver_position_iterations": args.solver_position_iterations,
    }
    args.output.with_suffix(".physics.json").write_text(json.dumps(report, indent=2) + "\n")
    conversion_path = args.usd.with_suffix(".conversion.json")
    if conversion_path.exists():
        conversion = json.loads(conversion_path.read_text())
        conversion.update(usd=str(args.output), physics_configured=True, mass_kg=args.mass, physics_report=str(args.output.with_suffix(".physics.json")))
        args.output.with_suffix(".conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
