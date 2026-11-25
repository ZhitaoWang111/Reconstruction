"""Verify an articulated physics USD in Isaac Sim: it loads as one articulation,
the root holds still, and every movable joint swings and stays inside its limits.

Checks, in order, each recorded in the JSON report:

* the stage has one articulation root and the expected bodies and joints;
* runtime masses match the authored ones;
* at rest (``--settle`` seconds, gravity on) the root body does not move more than
  5 mm (static) or comes to rest on the ground (dynamic), and no joint drifts by
  more than 2 degrees (2 mm for prismatic);
* each movable joint in turn is pushed open with a real joint effort (a torque or
  force held for ``--push-seconds``, then released): it must open by at least a
  third of its range and never leave its limits by more than 1 degree, while the
  root stays put (static) or upright (dynamic).  Setting joint velocities instead
  would override the solver every step and prove nothing about the physics.

Run inside the Isaac Sim environment::

    source /home/seer/wzt/robot_sim/scripts/activate_env.sh
    python -u src/objgen/dualcraft/check_articulated_isaacsim.py physics.usdc --output <dir>
"""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--settle", type=float, default=2.0, help="seconds of rest before kicking joints")
parser.add_argument("--push-seconds", type=float, default=0.5, help="how long each joint is pushed")
parser.add_argument("--release-seconds", type=float, default=2.0, help="seconds simulated after release")
parser.add_argument("--push-scale", type=float, default=3.0,
                    help="push effort per joint: this many times the effort that would hold the moving "
                         "body's own weight one body-length out (N*m for revolute, N for prismatic)")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
report = {"passed": False, "asset": str(args.usd.resolve())}
report_path = args.output / "isaacsim_articulation_verification.json"

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
try:
    from importlib.metadata import version

    import numpy as np
    import omni.usd
    from pxr import UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation

    physics_dt = 1 / 120
    assert omni.usd.get_context().open_stage(str(args.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI), "default prim is not an articulation root"
    authored_mass = {p.GetName(): float(UsdPhysics.MassAPI(p).GetMassAttr().Get())
                     for p in root.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    fixed_base = bool(stage.GetPrimAtPath(root.GetPath().AppendPath("joints/fix_base")))

    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=1 / 60)
    world.scene.add_default_ground_plane()
    cabinet = world.scene.add(SingleArticulation(str(root.GetPath()), name="asset"))
    world.reset()
    names = list(cabinet.dof_names)
    bodies = list(cabinet._articulation_view.body_names)
    limits = np.asarray(cabinet._articulation_view.get_dof_limits())[0]
    runtime_mass = dict(zip(bodies, map(float, np.asarray(cabinet._articulation_view.get_body_masses())[0])))
    report.update(isaacsim=version("isaacsim"), dof_names=names, bodies=bodies, fixed_base=fixed_base,
                  limits=limits.tolist(), authored_mass_kg=authored_mass, runtime_mass_kg=runtime_mass)
    checks = {"one_articulation_with_all_bodies": sorted(bodies) == sorted(authored_mass),
              "runtime_mass_matches": all(abs(runtime_mass.get(n, -1) - m) < 1e-3 * max(1.0, m)
                                          for n, m in authored_mass.items())}
    revolute = [stage.GetPrimAtPath(root.GetPath().AppendPath(f"joints/{n}")).IsA(UsdPhysics.RevoluteJoint)
                for n in names]
    tolerance = [math.radians(2.0) if r else 0.002 for r in revolute]

    def root_pose():
        position, _ = cabinet._articulation_view.get_world_poses()
        return np.asarray(position)[0]

    def root_tilt_deg():
        _, orientation = cabinet._articulation_view.get_world_poses()
        w, x, y, z = np.asarray(orientation)[0]
        up_z = 1 - 2 * (x * x + y * y)          # world-Z component of the body's Z axis
        return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))

    start_root = root_pose().copy()
    q0 = np.asarray(cabinet.get_joint_positions()).copy()
    for _ in range(int(args.settle / physics_dt)):
        world.step(render=False)
    q_rest = np.asarray(cabinet.get_joint_positions())
    drift = np.abs(q_rest - q0)
    root_shift = float(np.linalg.norm(root_pose() - start_root))
    tilt = root_tilt_deg()
    root_speed = float(np.linalg.norm(np.asarray(cabinet._articulation_view.get_velocities())[0][:3]))
    report["settle"] = {"seconds": args.settle, "joint_drift": dict(zip(names, drift.tolist())),
                        "root_shift_m": root_shift, "root_tilt_deg": tilt, "root_speed_m_s": root_speed}
    checks["joints_stay_closed_at_rest"] = bool(np.all(drift <= tolerance))
    if fixed_base:
        checks["root_holds_at_rest"] = root_shift < 0.005
    else:
        # Free-standing: authored standing on z = 0, so it should stay put, upright and still.
        checks["root_stands_at_rest"] = bool(root_shift < 0.02 and tilt < 2.0 and root_speed < 0.05)

    # Effort scale per joint from the authored masses: g * m * r for a revolute joint
    # (r = distance from the axis to the body's centre of mass), g * m for prismatic.
    effort_scale = []
    for index, name in enumerate(names):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(root.GetPath().AppendPath(f"joints/{name}")))
        child = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        mass = float(UsdPhysics.MassAPI(child).GetMassAttr().Get())
        com = np.asarray(UsdPhysics.MassAPI(child).GetCenterOfMassAttr().Get(), dtype=float)
        pos1 = np.asarray(joint.GetLocalPos1Attr().Get(), dtype=float)
        lever = float(np.linalg.norm((com - pos1)[:2])) if revolute[index] else 1.0
        effort_scale.append(9.81 * mass * max(lever, 0.05))

    kicks = {}
    for index, name in enumerate(names):
        cabinet.set_joint_positions(q_rest.copy())
        cabinet.set_joint_velocities(np.zeros(len(names)))
        world.step(render=False)
        lower, upper = limits[index]
        direction = 1.0 if abs(upper) >= abs(lower) else -1.0
        effort = direction * args.push_scale * effort_scale[index]
        trace = []
        root_start = root_pose().copy()
        push_steps = int(args.push_seconds / physics_dt)
        for step in range(push_steps + int(args.release_seconds / physics_dt)):
            efforts = np.zeros(len(names))
            if step < push_steps:
                efforts[index] = effort
            cabinet.set_joint_efforts(efforts)
            world.step(render=False)
            trace.append(float(np.asarray(cabinet.get_joint_positions())[index]))
        cabinet.set_joint_efforts(np.zeros(len(names)))
        trace = np.array(trace)
        slack = math.radians(1.0) if revolute[index] else 0.001
        opened = float(np.max(np.abs(trace - q_rest[index])))
        kicks[name] = {
            "range": [float(lower), float(upper)], "push_effort": effort,
            "max_opening": opened, "final": float(trace[-1]),
            "min": float(trace.min()), "max": float(trace.max()),
            "within_limits": bool(trace.min() >= lower - slack and trace.max() <= upper + slack),
            "opened_a_third": bool(opened >= abs(upper - lower) / 3),
            "root_shift_m": float(np.linalg.norm(root_pose() - root_start)),
        }
    report["kicks"] = kicks
    checks["every_joint_opens"] = all(k["opened_a_third"] for k in kicks.values())
    checks["every_joint_within_limits"] = all(k["within_limits"] for k in kicks.values())
    if fixed_base:
        checks["root_holds_during_kicks"] = all(k["root_shift_m"] < 0.005 for k in kicks.values())
    else:
        checks["root_upright_after_kicks"] = bool(root_tilt_deg() < 5.0)
    world.stop()
    report.update(checks=checks, passed=all(checks.values()), physics_dt_s=physics_dt)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("ARTICULATION_VERIFIED" if report["passed"] else "ARTICULATION_FAILED", json.dumps(checks), flush=True)
except BaseException as exc:
    import traceback
    report.update(passed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("ARTICULATION_ERROR", report["error"], flush=True)
finally:
    app.close()
