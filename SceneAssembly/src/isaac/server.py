"""Isaac Sim service: renders and gravity runs for scene assembly, in one long-lived process.

Runs under the standalone Isaac Sim's Python, not SceneAssembly/.venv:

    ~/isaacsim/python.sh src/isaac/server.py --socket /tmp/x.sock --authkey-env ISAAC_SERVICE_KEY

Normally started and talked to through `isaac.client.IsaacService`. The harness
owns the scene state: every request carries the full set of objects that should
exist (oid -> asset USD + 4x4 pose) and the service makes the stage match it.
Objects not in the request are deactivated, so they neither render nor collide.
The walls and floors of the structure handoff are always loaded.

The service measures and reports; it never judges (thresholds live in the
harness) and never changes the harness's poses: after a gravity run the
objects are put back exactly where the request placed them.

Protocol: multiprocessing.connection over a Unix socket, one JSON object per
message, {"id", "op", ...} -> {"id", "ok", ...}. Ops: ping, render, settle,
raycast, shutdown. Isaac's own logging goes to stdout/stderr, which the client
sends to a log file, so it can never mix with replies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--socket', required=True)
ap.add_argument('--authkey-env', default='ISAAC_SERVICE_KEY')
ap.add_argument('--structure', required=True, help='structure handoff USD (walls, floors)')
args = ap.parse_args()

from isaacsim import SimulationApp

# fast_shutdown: without it this kit build was seen to hang after every extension had shut down.
app = SimulationApp({'headless': True, 'width': 1296, 'height': 968, 'fast_shutdown': True})

import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from omni.physx import get_physx_interface, get_physx_scene_query_interface, get_physx_simulation_interface
from PIL import Image
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

# run as a script: put src/ (not this directory) first on the path, so `isaac.camera` resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaac.camera import gf_matrix, set_intrinsics, usd_camera_to_world  # noqa: E402

PHYSICS_HZ = 120
SAMPLE_EVERY = 12            # steps between trajectory samples: 0.1 s
SETTLED_WINDOW_S = 0.5       # "settled" looks at the last half second
SETTLED_SPEED = 0.01         # m/s, and the same number in rad/s, over that window
EARLY_S = 0.1                # early_jump: displacement within the first 0.1 s
OBJECTS = '/World/Objects'


def log(*a):
    print('[isaac_server]', *a, flush=True)


# ---------------------------------------------------------------- stage

ctx = omni.usd.get_context()
ctx.new_stage()
for _ in range(3):
    app.update()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, '/World')
stage.SetDefaultPrim(world.GetPrim())
UsdGeom.Xform.Define(stage, OBJECTS)
stage.DefinePrim('/World/Structure', 'Xform').GetReferences().AddReference(str(Path(args.structure).resolve()))

scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
px_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px_scene.CreateEnableGPUDynamicsAttr(False)          # CPU: small scenes, and repeatable
px_scene.CreateEnableEnhancedDeterminismAttr(True)
px_scene.CreateBroadphaseTypeAttr('MBP')
px_scene.CreateTimeStepsPerSecondAttr(PHYSICS_HZ)

UsdLux.DomeLight.Define(stage, '/World/DomeLight').CreateIntensityAttr(1000.0)
camera = UsdGeom.Camera.Define(stage, '/World/ServiceCamera')
camera_op = camera.AddTransformOp()
for _ in range(10):
    app.update()

timeline = omni.timeline.get_timeline_interface()
physx = get_physx_interface()
physx_sim = get_physx_simulation_interface()
rep.orchestrator.set_capture_on_play(False)

# oid -> {"usd": str, "matrix": tuple} as last applied; lets a request touch only what changed
applied: dict = {}


def object_path(oid: str) -> str:
    return f'{OBJECTS}/{oid}'


def apply_objects(objects: dict, force_pose=False):
    """Make the stage hold exactly `objects` ({oid: {"usd", "matrix"}}) and nothing else."""
    for oid, spec in objects.items():
        prim = stage.GetPrimAtPath(object_path(oid))
        if not prim.IsValid():
            x = UsdGeom.Xform.Define(stage, object_path(oid))
            x.AddTransformOp()
            prim = x.GetPrim()
            prim.GetReferences().AddReference(spec['usd'])
        elif applied.get(oid, {}).get('usd') != spec['usd']:
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddReference(spec['usd'])
        if not prim.IsActive():
            prim.SetActive(True)
        m = tuple(map(float, np.asarray(spec['matrix'], float).reshape(-1)))
        if force_pose or applied.get(oid, {}).get('matrix') != m:
            UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(gf_matrix(np.array(m).reshape(4, 4)))
        applied[oid] = {'usd': spec['usd'], 'matrix': m}
    for prim in stage.GetPrimAtPath(OBJECTS).GetChildren():
        if prim.GetName() not in objects and prim.IsActive():
            prim.SetActive(False)
            applied.pop(prim.GetName(), None)
    app.update()


def body_kind(oid: str) -> str:
    """dynamic | kinematic | static, as the asset authored it."""
    prim = stage.GetPrimAtPath(object_path(oid))
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return 'static'
    rb = UsdPhysics.RigidBodyAPI(prim)
    if rb.GetRigidBodyEnabledAttr().Get() is False:
        return 'static'
    return 'kinematic' if rb.GetKinematicEnabledAttr().Get() else 'dynamic'


# ---------------------------------------------------------------- render

render_product = None
render_size = None
annotators: dict = {}
ANNOTATORS = {'rgb': ('rgb', {}),
              'depth': ('distance_to_image_plane', {}),
              'labels': ('instance_id_segmentation_fast', {'init_params': {'colorize': False}})}


def ensure_render_product(width, height, outputs):
    global render_product, render_size, annotators
    if render_product is not None and render_size != (width, height):
        for a in annotators.values():
            a.detach()
        render_product.destroy()
        render_product, annotators = None, {}
    if render_product is None:
        render_product = rep.create.render_product(str(camera.GetPath()), (width, height))
        render_size = (width, height)
    for name in outputs:
        if name not in annotators:
            kind, kw = ANNOTATORS[name]
            a = rep.AnnotatorRegistry.get_annotator(kind, **kw)
            a.attach(render_product)
            annotators[name] = a


def label_of(path: str) -> str:
    if path.startswith(OBJECTS + '/'):
        return path[len(OBJECTS) + 1:].split('/')[0]
    if path.startswith('/World/Structure/Walls/'):
        return 'wall'
    if path.startswith('/World/Structure/Floors/'):
        return 'floor'
    return 'other'


def op_render(req):
    cam = req['camera']
    W, H = int(cam['width']), int(cam['height'])
    outputs = req.get('outputs', ['rgb'])
    subframes = int(req.get('subframes', 4))
    out = Path(req['out'])
    out.mkdir(parents=True, exist_ok=True)
    t_apply = time.monotonic()
    apply_objects(req['objects'])
    apply_ms = round((time.monotonic() - t_apply) * 1000)
    set_intrinsics(camera, cam['fx'], cam['fy'], cam['cx'], cam['cy'], W, H)
    ensure_render_product(W, H, outputs)
    results = []
    for view in req['views']:
        t = time.monotonic()
        camera_op.Set(gf_matrix(usd_camera_to_world(view['c2w'])))
        rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0, pause_timeline=True)
        t_render = time.monotonic()
        r = {'name': view['name']}
        rgb_path = out / f"{view['name']}.png"
        Image.fromarray(np.asarray(annotators['rgb'].get_data())[..., :3]).save(rgb_path, compress_level=1)
        r['rgb'] = str(rgb_path)
        extra = {}
        if 'depth' in outputs:
            d = np.asarray(annotators['depth'].get_data(), np.float32)
            extra['depth'] = np.where(np.isfinite(d) & (d > 0), d, np.inf).astype(np.float32)
        if 'labels' in outputs:
            seg = annotators['labels'].get_data()
            ids = np.asarray(seg['data']).astype(np.int64)
            names = ['none']
            table = np.zeros(int(ids.max()) + 1 if ids.size else 1, np.int16)
            for k, path in seg['info']['idToLabels'].items():
                k = int(k)
                if k >= len(table):
                    continue
                lab = label_of(str(path))
                if lab not in names:
                    names.append(lab)
                table[k] = names.index(lab)
            idx = table[ids]
            extra['labels'], extra['label_names'] = idx, np.array(names)
            counts = np.bincount(idx.ravel(), minlength=len(names))
            r['visible_px'] = {n: int(c) for n, c in zip(names, counts) if n != 'none' and c}
        if extra:
            p = out / f"{view['name']}.npz"
            np.savez(p, **extra)
            r['arrays'] = str(p)
        r['render_ms'] = round((t_render - t) * 1000)
        r['ms'] = round((time.monotonic() - t) * 1000)
        results.append(r)
    return {'views': results, 'apply_ms': apply_ms}


# ---------------------------------------------------------------- physics

def rotation_of(m4: np.ndarray) -> np.ndarray:
    """Rotation part of a 4x4 that may carry scale (polar decomposition)."""
    u, _, vt = np.linalg.svd(m4[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_body(oid):
    t = physx.get_rigidbody_transformation(object_path(oid))
    if not t.get('ret_val'):
        raise RuntimeError(f'PhysX has no rigid body for {oid}')
    return np.array(t['position'], float), quat_xyzw_to_matrix(t['rotation'])


def angle_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def op_settle(req):
    objects = req['objects']
    seconds = float(req.get('seconds', 5.0))
    movable_req = list(req['movable'])
    apply_objects(objects, force_pose=True)
    kinds = {oid: body_kind(oid) for oid in objects}
    fixed = [o for o in movable_req if o in kinds and kinds[o] != 'dynamic']
    unknown = [o for o in movable_req if o not in objects]
    movable = [o for o in movable_req if kinds.get(o) == 'dynamic']
    # every other dynamic body is held where it is for this run
    held = [o for o, k in kinds.items() if k == 'dynamic' and o not in movable]
    for o in held:
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(object_path(o))).GetKinematicEnabledAttr().Set(True)
    start = {}
    for o in movable:
        m = np.array(applied[o]['matrix']).reshape(4, 4)
        start[o] = (m[:3, 3].copy(), rotation_of(m))
    report = {}
    t0 = time.monotonic()
    try:
        timeline.play()
        app.update()                       # parses the stage into PhysX; also advances one frame
        steps = int(round(seconds * PHYSICS_HZ))
        dt = 1.0 / PHYSICS_HZ
        traj = {o: [] for o in movable}    # (t, position, rotation) every SAMPLE_EVERY steps
        for o in movable:
            traj[o].append((0.0, *start[o]))
        for i in range(steps):
            physx_sim.simulate(dt, i * dt)
            physx_sim.fetch_results()
            if (i + 1) % SAMPLE_EVERY == 0 or i + 1 == steps:
                for o in movable:
                    traj[o].append(((i + 1) * dt, *read_body(o)))
        for o in movable:
            p0, r0 = start[o]
            ts = np.array([s[0] for s in traj[o]])
            ps = np.array([s[1] for s in traj[o]])
            p1, r1 = traj[o][-1][1], traj[o][-1][2]
            up_now = (r1 @ r0.T) @ np.array([0, 0, 1.0])
            early = ps[ts <= EARLY_S + 1e-9]
            tail = ts >= ts[-1] - SETTLED_WINDOW_S - 1e-9
            k = np.flatnonzero(tail)
            span = max(ts[k[-1]] - ts[k[0]], 1e-6)
            speed = float(np.linalg.norm(ps[k[-1]] - ps[k[0]]) / span)
            spin = math.radians(angle_deg(traj[o][k[-1]][2] @ traj[o][k[0]][2].T)) / span
            m_end = np.eye(4)
            m_end[:3, :3], m_end[:3, 3] = r1, p1
            report[o] = {
                'dxy': round(float(np.linalg.norm((p1 - p0)[:2])), 4),
                'dz': round(float(p1[2] - p0[2]), 4),
                'tilt_deg': round(math.degrees(math.acos(max(-1.0, min(1.0, up_now[2])))), 2),
                'turn_deg': round(angle_deg(r1 @ r0.T), 2),
                'early_jump': round(float(np.linalg.norm(early - p0, axis=1).max()), 4),
                'settled': bool(speed < SETTLED_SPEED and spin < SETTLED_SPEED),
                'end_matrix': np.round(m_end, 5).tolist(),
                'track_t': np.round(ts, 3).tolist(),
                'track_z': np.round(ps[:, 2], 4).tolist(),
            }
    finally:
        timeline.stop()
        app.update()
        for o in held:   # drop the override; the asset's own value (dynamic) shows through again
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(object_path(o))).GetKinematicEnabledAttr().Clear()
        apply_objects(objects, force_pose=True)   # put every pose back exactly as requested
    return {'moved': report, 'fixed': fixed, 'unknown': unknown, 'held': held,
            'seconds': seconds, 'hz': PHYSICS_HZ, 'ms': round((time.monotonic() - t0) * 1000)}


def op_contacts(req):
    """Which objects are in contact, at the poses given. No distance (M4 decision, 2026-11-20).

    One physics step with gravity off, so nothing moves; then the contact report.
    Bodies must stay dynamic: PhysX reports no contacts between two kinematic bodies.
    """
    objects = req['objects']
    apply_objects(objects, force_pose=True)
    scene = UsdPhysics.Scene(stage.GetPrimAtPath('/World/PhysicsScene'))
    gravity = scene.GetGravityMagnitudeAttr().Get()
    scene.GetGravityMagnitudeAttr().Set(0.0)
    tagged = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
            tagged.append(prim)
    pairs = {}
    t0 = time.monotonic()
    try:
        timeline.play()
        app.update()                       # parse the stage into PhysX
        physx_sim.simulate(1.0 / PHYSICS_HZ, 0.0)
        physx_sim.fetch_results()
        heads, _ = physx_sim.get_contact_report()
        for h in heads:
            if h.num_contact_data == 0:
                continue
            a = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
            b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            key = tuple(sorted((a, b)))
            pairs[key] = pairs.get(key, 0) + int(h.num_contact_data)
    finally:
        timeline.stop()
        app.update()
        scene.GetGravityMagnitudeAttr().Set(gravity)
        for prim in tagged:
            prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
        apply_objects(objects, force_pose=True)
    out = []
    for (a, b), n in pairs.items():
        out.append({'a': label_of(a), 'b': label_of(b), 'a_path': a, 'b_path': b, 'contacts': n})
    return {'pairs': out, 'ms': round((time.monotonic() - t0) * 1000)}


def op_raycast(req):
    """Which collider shapes a ray hits, in physics as it is parsed. Diagnostic."""
    apply_objects(req['objects'], force_pose=True)
    hits = []

    def on_hit(h):
        hits.append({'collision': str(h.collision), 'rigid_body': str(h.rigid_body), 'distance': round(float(h.distance), 4)})
        return True

    try:
        timeline.play()
        app.update()
        get_physx_scene_query_interface().raycast_all(
            Gf.Vec3f(*map(float, req['origin'])), Gf.Vec3f(*map(float, req['direction'])), float(req.get('distance', 10.0)), on_hit)
    finally:
        timeline.stop()
        app.update()
        apply_objects(req['objects'], force_pose=True)
    return {'hits': sorted(hits, key=lambda h: h['distance'])}


# ---------------------------------------------------------------- loop

OPS = {'ping': lambda req: {'pong': True}, 'render': op_render, 'settle': op_settle,
       'contacts': op_contacts, 'raycast': op_raycast}

key = os.environ[args.authkey_env].encode()
sock = Path(args.socket)
if sock.exists():
    sock.unlink()
listener = Listener(str(sock), family='AF_UNIX', authkey=key)
log('READY', sock)
try:
    running = True
    while running:
        conn = listener.accept()
        try:
            while True:
                try:
                    raw = conn.recv_bytes()
                except EOFError:
                    break
                req = json.loads(raw)
                rid, op = req.get('id'), req.get('op')
                if op == 'shutdown':
                    conn.send_bytes(json.dumps({'id': rid, 'ok': True}).encode())
                    running = False
                    break
                t = time.monotonic()
                try:
                    reply = {'id': rid, 'ok': True, **OPS[op](req)}
                except Exception as e:  # report and keep serving; the client decides what to do
                    log('ERROR', op, traceback.format_exc())
                    reply = {'id': rid, 'ok': False, 'error': f'{type(e).__name__}: {e}'}
                reply['service_ms'] = round((time.monotonic() - t) * 1000)
                conn.send_bytes(json.dumps(reply).encode())
        finally:
            conn.close()
finally:
    listener.close()
    if sock.exists():
        sock.unlink()
    log('EXIT')
    app.close()
