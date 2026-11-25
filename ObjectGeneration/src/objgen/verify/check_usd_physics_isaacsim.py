"""Verify an asset's authored physics and a drop onto a ground plane in Isaac Sim."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mass", type=float, default=None,
                    help="Optional expected mass in kilograms; otherwise use the asset's authored mass (never overrides it)")
parser.add_argument("--clearance-probe", action="store_true", help="Verify that an 8 cm cube can stay on the ground beneath the asset")
parser.add_argument("--steps", type=int, default=600,
                    help="Simulation steps at 120 Hz, at least 600; e.g. 1200 allows 10 seconds to settle")
args = parser.parse_args()
if args.mass is not None and (not math.isfinite(args.mass) or args.mass <= 0):
    parser.error("--mass must be a finite positive number in kilograms")
if args.steps < 600:
    parser.error("--steps must be at least 600")
args.output.mkdir(parents=True, exist_ok=True)
report = {"passed": False, "asset": str(args.usd.resolve()), "physics_steps_requested": args.steps}

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.prims import SingleRigidPrim

    assert omni.usd.get_context().open_stage(str(args.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    root = stage.GetDefaultPrim()
    mass_api = UsdPhysics.MassAPI(root)
    authored_mass = mass_api.GetMassAttr().Get()
    if authored_mass is None or not math.isfinite(authored_mass) or authored_mass <= 0:
        raise ValueError(f"Asset must have a finite positive authored mass; got {authored_mass!r}")
    authored_mass = float(authored_mass)
    if args.mass is not None and abs(authored_mass - args.mass) >= 1e-6:
        raise ValueError(f"Asset mass {authored_mass} kg does not match --mass {args.mass} kg; asset was not modified")
    assert UsdPhysics.RigidBodyAPI(root).GetRigidBodyEnabledAttr().Get()
    assert not UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
    mesh = stage.GetPrimAtPath(root.GetPath().AppendChild("Mesh"))
    assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get()
    approximation = UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get()
    assert approximation in ("convexHull", "convexDecomposition")
    local_points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
    def runtime_bottom(position, orientation):
        rotation = np.asarray(Gf.Matrix3d(1).SetRotate(Gf.Quatd(float(orientation[0]), Gf.Vec3d(*map(float, orientation[1:])))))
        return float((local_points @ rotation[:, 2]).min() + position[2])
    material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial("physics")[0]
    assert material
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 60)
    world.scene.add_default_ground_plane()
    bottle = world.scene.add(SingleRigidPrim(
        prim_path=str(root.GetPath()), name="bottle", position=np.array([0.0, 0.0, 0.20]),
    ))
    clearance_probe = None
    if args.clearance_probe:
        clearance_probe = world.scene.add(DynamicCuboid(
            prim_path="/PhysicsClearanceProbe", name="clearance_probe",
            position=np.array([0.0, 0.0, 0.04]), size=0.08, mass=0.1,
        ))
    world.reset()
    mass_kg = float(bottle.get_mass())
    if not math.isfinite(mass_kg) or mass_kg <= 0 or abs(mass_kg - authored_mass) >= 1e-6:
        raise ValueError(f"Runtime mass {mass_kg} kg does not match asset mass {authored_mass} kg")
    initial_position, _ = bottle.get_world_pose()
    initial_position = initial_position.copy()
    minimum_bottom = float("inf")
    positions = []
    for step in range(args.steps):
        world.step(render=False)
        position, orientation = bottle.get_world_pose()
        assert np.isfinite(position).all() and np.isfinite(orientation).all()
        # Tensor poses remain current even when render=False does not write transforms to USD.
        minimum_bottom = min(minimum_bottom, runtime_bottom(position, orientation))
        if step in (0, 29, 59, 119, 299, 599) or (step + 1) % 120 == 0 or step == args.steps - 1:
            positions.append({"step": step + 1, "position_m": position.tolist(),
                              "linear_speed_m_s": float(np.linalg.norm(bottle.get_linear_velocity())),
                              "angular_speed_rad_s": float(np.linalg.norm(bottle.get_angular_velocity()))})
    final_position, final_orientation = bottle.get_world_pose()
    velocity = bottle.get_linear_velocity()
    angular_velocity = bottle.get_angular_velocity()
    final_bottom = runtime_bottom(final_position, final_orientation)
    checks = {
        "dropped_at_least_10cm": bool(final_position[2] < initial_position[2] - 0.10),
        "no_deep_ground_penetration": bool(minimum_bottom > -0.01),
        "linear_speed_below_0_05": bool(np.linalg.norm(velocity) < 0.05),
        "angular_speed_below_0_10": bool(np.linalg.norm(angular_velocity) < 0.10),
        "final_bottom_near_ground": bool(-0.005 < final_bottom < 0.015),
    }
    clearance_position = None
    if clearance_probe is not None:
        clearance_position, _ = clearance_probe.get_world_pose()
        checks["clearance_probe_stays_on_ground"] = bool(abs(clearance_position[2] - 0.04) < 0.015)
        checks["clearance_probe_stays_under_asset"] = bool(np.linalg.norm(clearance_position[:2]) < 0.02)
    world.stop()
    report = {
        "passed": all(checks.values()), "checks": checks,
        "isaacsim": version("isaacsim"), "asset": str(args.usd.resolve()),
        "authored_mass_kg": authored_mass, "expected_mass_kg": args.mass,
        "runtime_mass_kg": mass_kg, "physics_steps": args.steps, "physics_dt_s": 1 / 120,
        "simulation_seconds": args.steps / 120,
        "collision_approximation": approximation,
        "initial_position_m": initial_position.tolist(), "final_position_m": final_position.tolist(),
        "minimum_bottom_z_m": minimum_bottom, "final_bottom_z_m": final_bottom,
        "final_linear_velocity_m_s": velocity.tolist(), "final_angular_velocity_rad_s": angular_velocity.tolist(),
        "final_linear_speed_m_s": float(np.linalg.norm(velocity)),
        "final_angular_speed_rad_s": float(np.linalg.norm(angular_velocity)),
        "position_samples": positions, "drop_and_ground_collision_verified": all(checks.values()),
        "under_asset_clearance_verified": args.clearance_probe and all(checks.values()),
        "clearance_probe_position_m": clearance_position.tolist() if clearance_position is not None else None,
    }
    if not report["passed"]:
        raise RuntimeError("Physics checks failed: " + ", ".join(name for name, passed in checks.items() if not passed))
    (args.output / "isaacsim_physics_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("USD_PHYSICS_VERIFIED", json.dumps(report), flush=True)
except BaseException as exc:
    report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    (args.output / "isaacsim_physics_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    traceback.print_exc()
    print("USD_PHYSICS_FAILED", flush=True)
    raise
finally:
    app.close()
