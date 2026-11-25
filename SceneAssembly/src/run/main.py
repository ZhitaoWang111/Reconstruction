"""Run all batches in order, resumable; final physics; export. See run/__init__.py."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from inputs import handoff, structure as structure_mod          # noqa: E402
from initial import InitialPoses                                # noqa: E402
from state import SceneState                                    # noqa: E402
from verify import Config, Verifier                             # noqa: E402
from observe import Observer                                    # noqa: E402
from batch import cut_batches, BatchSession                     # noqa: E402
from batch.plan import write_plan                               # noqa: E402

RUNS = HERE.parents[1] / 'runs'
FINAL_PHYSICS_S = 15.0
DEFAULT_MAX_USD = 150.0


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Run:
    def __init__(self, name: str, max_usd: float = DEFAULT_MAX_USD, effort: str = 'high'):
        self.dir = RUNS / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.max_usd = max_usd
        self.effort = effort
        self.scene = handoff.load()
        self.structure = structure_mod.load(scene=self.scene)
        self.plan = cut_batches(self.scene)
        self.cfg = Config()
        self.progress_path = self.dir / 'progress.json'
        self.progress = json.loads(self.progress_path.read_text()) if self.progress_path.exists() else {
            'batches': {}, 'usd': 0.0, 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
        self._write_config()

    # ------------------------------------------------------------ bookkeeping

    def _write_config(self):
        p = self.dir / 'config.json'
        hand = handoff.ASSET_HANDOFF
        cfg = {'name': self.name, 'scene_id': self.scene.scene_id, 'frame': self.scene.frame,
               'asset_handoff': str(hand), 'asset_manifest_sha256': _sha(Path(hand) / 'manifest.json'),
               'structure_handoff': str(self.structure.root),
               'structure_manifest_sha256': _sha(self.structure.root / 'manifest.json'),
               'verify': self.cfg.to_dict(), 'max_usd': self.max_usd, 'effort': self.effort,
               'batches': [b.to_dict() for b in self.plan]}
        from agent.llm import MODEL
        cfg['model'] = MODEL
        if p.exists():
            old = json.loads(p.read_text())
            for k in ('asset_manifest_sha256', 'structure_manifest_sha256', 'batches'):
                if old.get(k) != cfg[k]:
                    raise SystemExit(f'{p}: {k} changed since this run started; start a new run instead of resuming')
        else:
            p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
        write_plan(self.plan, self.dir / 'plan.json')

    def _save_progress(self):
        self.progress_path.write_text(json.dumps(self.progress, indent=1, ensure_ascii=False))

    def checkpoints(self) -> dict:
        p = self.dir / 'checkpoints.json'
        return {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}

    # ------------------------------------------------------------ state

    def load_state(self, before_batch: int):
        """The state as it was when `before_batch` should start: the latest checkpoint of an
        earlier batch. Snapshots written after it (an interrupted batch) are moved aside."""
        states = self.dir / 'state' / 'states'
        cps = {b: v for b, v in self.checkpoints().items() if b < before_batch}
        if not states.exists() or not any(states.glob('v*.json')):
            return SceneState(self.scene, self.structure, self.dir / 'state'), None
        version = max(cps.values()) if cps else None
        if version is None:
            # nothing finished yet: start clean, keep what was there for the record
            self._archive(states, keep_upto=0)
            return SceneState(self.scene, self.structure, self.dir / 'state'), None
        self._archive(states, keep_upto=version)
        return SceneState.load(self.scene, self.structure, self.dir / 'state', version=version), version

    def _archive(self, states: Path, keep_upto: int):
        later = [f for f in states.glob('v*.json') if int(f.stem[1:]) > keep_upto]
        if not later:
            return
        dest = self.dir / 'state' / f'discarded_{time.strftime("%Y%m%d_%H%M%S")}'
        dest.mkdir(parents=True)
        for f in later:
            shutil.move(str(f), dest / f.name)
        # the action log is kept whole; a marker shows where the resume cut it
        with open(self.dir / 'state' / 'actions.jsonl', 'a') as f:
            f.write(json.dumps({'resume': True, 'kept_upto': keep_upto, 'moved_to': dest.name}) + '\n')

    # ------------------------------------------------------------ run

    def batches_to_run(self, only=None) -> list:
        done = {int(b) for b, r in self.progress['batches'].items() if r.get('closed')}
        idx = [b.index for b in self.plan if b.index not in done]
        if only:
            idx = [i for i in idx if i in set(only)]
        return idx

    def run(self, only=None, fake_client=None):
        from isaac.client import IsaacService
        from agent.loop import Context, run_batch
        import agent.loop as L
        if fake_client is not None:
            L.Client = fake_client
        todo = self.batches_to_run(only)
        if not todo:
            print('nothing left to run')
        with IsaacService(self.structure.usd, log_path=self.dir / 'isaac_service.log') as isaac:
            for index in todo:
                if self.progress['usd'] >= self.max_usd:
                    self.progress['stopped'] = f'cost cap ${self.max_usd} reached before batch {index}'
                    self._save_progress()
                    print(self.progress['stopped'])
                    return self.progress
                state, version = self.load_state(index)
                initial = InitialPoses(self.scene, self.structure, self.dir / 'init')
                initial.compute_all()
                observer = Observer(self.scene, self.structure, state, initial, isaac, self.dir / 'observe')
                verifier = Verifier(state, self.structure, isaac, self.cfg, log_dir=self.dir / 'verify')
                session = BatchSession(self.plan, self.scene, self.structure, state, initial, observer, verifier, self.dir)
                ctx = Context(self.scene, self.structure, state, initial, observer, verifier, isaac, self.plan)
                self.progress['batches'][str(index)] = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                        'from_version': version, 'closed': False}
                self._save_progress()
                summary = run_batch(index, ctx, self.dir, session=session, effort=self.effort)
                self.progress['usd'] = round(self.progress['usd'] + summary['usd'], 2)
                self.progress['batches'][str(index)].update(summary, closed=True)
                self._save_progress()
                print(f"batch {index}: {summary['stop']}, {summary['turns']} turns, ${summary['usd']}, "
                      f"objects {summary['objects']}", flush=True)
            if not self.batches_to_run():
                self.finish(isaac)
        return self.progress

    def finish(self, isaac):
        """After the last batch: one whole-scene gravity run (report and mark failed, no undo), then export."""
        state, version = self.load_state(10 ** 6)
        initial = InitialPoses(self.scene, self.structure, self.dir / 'init')
        verifier = Verifier(state, self.structure, isaac, self.cfg, log_dir=self.dir / 'verify')
        ph = verifier.review_physics(seconds=FINAL_PHYSICS_S, undo=False)
        for oid in ph['stop']:
            state.set_status(oid, 'failed', note='final physics: fell over or dropped')
        (self.dir / 'final_physics.json').write_text(json.dumps(ph, indent=1, ensure_ascii=False, default=str))
        out = self.export(state)
        self.progress['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        self.progress['final'] = {'state_version': state.version, 'physics_failed': ph['stop'], 'scene': str(out),
                                  'objects': {s: sum(1 for o in state.objects.values() if o.status == s)
                                              for s in ('done', 'failed', 'working', 'untouched')}}
        self._save_progress()
        print('finished:', json.dumps(self.progress['final'], ensure_ascii=False))

    def export(self, state=None):
        """Export the latest state. Reads only: unlike load_state, nothing is moved aside."""
        from run.export import export_scene
        if state is None:
            state = SceneState.load(self.scene, self.structure, self.dir / 'state')
        return export_scene(state, self.scene, self.structure, self.dir / 'export' / 'scene.usda', run_name=self.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--batches', type=int, nargs='*', help='only these batches (testing)')
    ap.add_argument('--max-usd', type=float, default=DEFAULT_MAX_USD)
    ap.add_argument('--effort', default='high')
    ap.add_argument('--export-only', action='store_true')
    a = ap.parse_args()
    r = Run(a.name, max_usd=a.max_usd, effort=a.effort)
    if a.export_only:
        print(r.export())
        return
    r.run(only=a.batches)


if __name__ == '__main__':
    main()
