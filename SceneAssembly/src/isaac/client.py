"""Harness-side handle on the Isaac Sim service (src/isaac/server.py).

    with IsaacService(structure_usd) as isaac:
        r = isaac.render(objects, views, camera, out_dir)
        s = isaac.settle(objects, movable=['obj_0016'], seconds=5)

`objects` is {oid: (asset_usd_path, 4x4 matrix)} for every object that should
exist right now; the service deactivates everything else. The harness stays the
only owner of poses, so a crashed service is restarted and the same request is
simply sent again.

The service runs under the standalone Isaac Sim (`ISAAC_SIM`, default
~/isaacsim) with a clean environment, so nothing from this venv, ROS or
robot_sim leaks into it. Its output goes to a log file, never to replies.
"""
from __future__ import annotations

import itertools
import json
import os
import secrets
import subprocess
import tempfile
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

ISAAC_SIM = Path(os.environ.get('ISAAC_SIM', Path.home() / 'isaacsim'))
SERVER = Path(__file__).resolve().parent / 'server.py'
STARTUP_TIMEOUT_S = 180       # first start on a fresh shader cache took ~60 s
TIMEOUTS_S = {'ping': 10, 'render': 30, 'settle': 60, 'contacts': 30, 'raycast': 30, 'shutdown': 10}


class ServiceError(RuntimeError):
    pass


class IsaacService:
    def __init__(self, structure_usd, log_path=None, isaac_sim=ISAAC_SIM):
        self.structure_usd = str(Path(structure_usd).resolve())
        self.isaac_sim = Path(isaac_sim)
        self.log_path = Path(log_path) if log_path else Path(tempfile.gettempdir()) / f'isaac_service_{os.getpid()}.log'
        self._ids = itertools.count(1)
        self.proc = None
        self.conn = None
        self.restarts = 0

    # ------------------------------------------------------------ lifecycle

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._key = secrets.token_hex(16)
        self._sock = str(Path(tempfile.gettempdir()) / f'isaac_service_{os.getpid()}_{secrets.token_hex(4)}.sock')
        env = {'HOME': os.environ['HOME'], 'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
               'OMNI_KIT_ACCEPT_EULA': 'YES', 'ACCEPT_EULA': 'Y', 'ISAAC_SERVICE_KEY': self._key}
        for k in ('CUDA_VISIBLE_DEVICES',):
            if k in os.environ:
                env[k] = os.environ[k]
        self._log = open(self.log_path, 'ab')
        self._log.write(f'\n==== start {time.strftime("%F %T")} ====\n'.encode())
        self._log.flush()
        self.proc = subprocess.Popen(
            [str(self.isaac_sim / 'python.sh'), str(SERVER), '--socket', self._sock,
             '--authkey-env', 'ISAAC_SERVICE_KEY', '--structure', self.structure_usd],
            cwd=str(self.isaac_sim), env=env, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True)
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise ServiceError(f'Isaac service exited during startup (code {self.proc.returncode}); see {self.log_path}')
            if Path(self._sock).exists():
                try:
                    self.conn = Client(self._sock, family='AF_UNIX', authkey=self._key.encode())
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    pass
            time.sleep(0.2)
        else:
            self.stop(force=True)
            raise ServiceError(f'Isaac service did not come up in {STARTUP_TIMEOUT_S} s; see {self.log_path}')
        self._call({'op': 'ping'})
        return self

    def stop(self, force=False):
        if self.conn is not None and not force:
            try:
                self._call({'op': 'shutdown'})
            except Exception:
                pass
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=20 if not force else 0.1)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
                self.proc.wait()
            self.proc = None
        if getattr(self, '_log', None):
            self._log.close()
            self._log = None

    def restart(self):
        self.stop(force=True)
        self.restarts += 1
        self.start()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------ calls

    def _call(self, req: dict) -> dict:
        req = {'id': next(self._ids), **req}
        timeout = TIMEOUTS_S[req['op']]
        self.conn.send_bytes(json.dumps(req).encode())
        if not self.conn.poll(timeout):
            raise TimeoutError(f"Isaac service: {req['op']} took longer than {timeout} s")
        reply = json.loads(self.conn.recv_bytes())
        if reply.get('id') != req['id']:
            raise ServiceError(f"reply id {reply.get('id')} for request {req['id']}")
        if not reply.get('ok'):
            raise ServiceError(f"{req['op']}: {reply.get('error')}")
        return reply

    def call(self, req: dict) -> dict:
        """Send a request; on a timeout or a dead service, restart once and send it again.
        Errors the service reports itself (bad request, PhysX error) are not retried."""
        try:
            return self._call(req)
        except (TimeoutError, EOFError, OSError, BrokenPipeError) as first:
            self.restart()
            try:
                return self._call(req)
            except Exception as second:
                raise ServiceError(f"{req['op']} failed twice: {first!r}; after restart: {second!r}") from second

    @staticmethod
    def _objects(objects: dict) -> dict:
        return {oid: {'usd': str(Path(usd).resolve()), 'matrix': np.asarray(m, float).tolist()}
                for oid, (usd, m) in objects.items()}

    def render(self, objects: dict, views: list, camera, out, outputs=('rgb',), subframes=4) -> dict:
        """views: [{"name": "seq_0045", "c2w": 4x4 OpenCV camera-to-world}, ...];
        camera: anything with fx, fy, cx, cy, width, height (handoff.Camera).
        outputs: any of "rgb", "depth", "labels" (rgb is always written)."""
        cam = {k: float(getattr(camera, k)) for k in ('fx', 'fy', 'cx', 'cy')}
        cam.update(width=int(camera.width), height=int(camera.height))
        return self.call({'op': 'render', 'objects': self._objects(objects), 'camera': cam,
                          'views': [{'name': v['name'], 'c2w': np.asarray(v['c2w'], float).tolist()} for v in views],
                          'out': str(Path(out).resolve()), 'outputs': sorted(set(outputs) | {'rgb'}),
                          'subframes': int(subframes)})

    def settle(self, objects: dict, movable, seconds=5.0) -> dict:
        """Gravity run. Only `movable` (dynamic assets) may move; every other dynamic body
        is held for the run, static and kinematic assets never move. Poses are restored
        afterwards; the reply carries per-object displacement, tilt and trajectory."""
        return self.call({'op': 'settle', 'objects': self._objects(objects), 'movable': list(movable),
                          'seconds': float(seconds)})

    def contacts(self, objects: dict) -> list:
        """Object pairs in contact at these poses, no distance. Wall/floor pairs come back with
        a='wall'/'floor'. PhysX reports contacts between colliders as it parsed them (hulls and
        decompositions), one step with gravity off, so nothing moves."""
        r = self.call({'op': 'contacts', 'objects': self._objects(objects)})
        return [p for p in r['pairs'] if p['a'] != p['b']]

    def raycast(self, objects: dict, origin, direction, distance=10.0) -> dict:
        return self.call({'op': 'raycast', 'objects': self._objects(objects), 'origin': list(map(float, origin)),
                          'direction': list(map(float, direction)), 'distance': float(distance)})
