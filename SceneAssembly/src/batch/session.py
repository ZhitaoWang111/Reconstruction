"""One batch from opening to checkpoint."""
from __future__ import annotations

import json
import time
from pathlib import Path

from state.resolve import Rejected

from .plan import Batch, first_key_frame

MAX_BROUGHT_IN = 3
MAX_REOPEN = 1
END_PHYSICS_S = 15.0


class BatchSession:
    def __init__(self, plan, scene, structure, state, initial, observer, verifier, run_dir: Path):
        self.plan = {b.index: b for b in plan}
        self.scene = scene
        self.structure = structure
        self.state = state
        self.initial = initial
        self.observer = observer
        self.verifier = verifier
        self.run = Path(run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        self.batch: Batch | None = None
        self.members: list = []            # objects of this batch, in processing order once set
        self.frames: list = []             # shared key frames
        self.brought_in: list = []         # objects brought in from other batches
        self.order_set = False
        self.reopened: dict = {}           # oid -> times reopened at the batch end
        self.moved_out: set = set()        # objects pulled forward into an earlier batch
        self.log: list = []
        self._load_moved_out()

    # ------------------------------------------------------------ bookkeeping

    def _note(self, event, **kw):
        e = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'batch': self.batch.index if self.batch else None,
             'event': event, **kw}
        self.log.append(e)
        with open(self.run / 'batches.jsonl', 'a') as f:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + '\n')
        return e

    def _load_moved_out(self):
        p = self.run / 'moved_out.json'
        if p.exists():
            self.moved_out = set(json.loads(p.read_text()))

    def _save_moved_out(self):
        (self.run / 'moved_out.json').write_text(json.dumps(sorted(self.moved_out)))

    def ensure_initialised(self):
        """Every object gets its starting pose once per run (M3), status untouched."""
        for oid in self.scene.objects:
            if oid not in self.state.objects:
                b = next(b.index for b in self.plan.values() if oid in b.objects)
                self.initial.apply(self.state, oid, batch=b)

    # ------------------------------------------------------------ opening

    def placed_summary(self) -> list:
        """What earlier batches left in the scene, as text (no images)."""
        out = []
        for oid, st in self.state.objects.items():
            if st.status in ('done', 'failed') and oid not in self.members:
                p = st.pose
                out.append({'object': oid, 'label': self.scene.objects[oid].label, 'status': st.status,
                            'x': round(p.x, 2), 'y': round(p.y, 2), 'yaw_deg': round(p.yaw_deg, 1),
                            'bottom_z': round(st.bottom_z, 3), 'support': st.support.to_dict()})
        return out

    def open(self, index: int) -> dict:
        self.ensure_initialised()
        b = self.plan[index]
        self.batch = b
        self.members = [o for o in b.objects if o not in self.moved_out]
        self.frames = sorted({first_key_frame(self.scene.objects[o]) for o in self.members})
        self.brought_in, self.order_set, self.reopened = [], False, {}
        self.state.current_batch = index
        for o in self.members:
            self.state.objects[o].batch = index        # stays untouched until started: not rendered yet (D04)
        pack = self.observer.batch_opening(self.members, self.frames)
        pack['placed'] = self.placed_summary()
        pack['batch'] = index
        pack['skipped'] = [o for o in b.objects if o in self.moved_out]
        (self.run / f'batch_{index:02d}').mkdir(exist_ok=True)
        (self.run / f'batch_{index:02d}' / 'opening.json').write_text(json.dumps(
            {'batch': index, 'objects': self.members, 'frames': self.frames, 'placed': pack['placed'],
             'skipped': pack['skipped']}, indent=1, ensure_ascii=False))
        self._note('open', objects=self.members, frames=self.frames, version=self.state.version)
        return pack

    # ------------------------------------------------------------ order

    def set_order(self, order: list, reason='') -> dict:
        """The agent's processing order. After the first time, only the unprocessed part may change."""
        if sorted(order) != sorted(self.members):
            raise Rejected(f'the order must list exactly this batch\'s objects: {sorted(self.members)}')
        done = [o for o in self.members if self.state.objects[o].status in ('done', 'failed')]
        if self.order_set and order[:len(done)] != [o for o in self.members if o in done]:
            raise Rejected(f'objects already processed keep their place; they are {done}')
        self.members = list(order)
        self.order_set = True
        return self._note('order', order=self.members, reason=reason)

    def processed(self) -> list:
        return [o for o in self.members if self.state.objects[o].status in ('done', 'failed')]

    def remaining(self) -> list:
        return [o for o in self.members if self.state.objects[o].status not in ('done', 'failed')]

    # ------------------------------------------------------------ one object

    def start(self, oid, note='') -> dict:
        """The agent turns to `oid`: from now on it is in the scene (renders, contacts)."""
        if oid not in self.members:
            raise Rejected(f'{oid} is not in this batch; bring it in first')
        if not self.order_set:
            raise Rejected('set the processing order first')
        if self.state.objects[oid].status == 'untouched':
            self.state.set_status(oid, 'working', note=note or 'started')
        return self._note('start', object=oid)

    def finish(self, oid, note='') -> dict:
        self.state.set_status(oid, 'done', note=note)
        return self._note('finish', object=oid, note=note)

    # ------------------------------------------------------------ bringing objects in

    def bring_in(self, oid, reason='') -> dict:
        """Bring an object from another batch into this one: an earlier one to fix it, or a later
        one that something here rests on. Its material comes with it."""
        if oid in self.members:
            raise Rejected(f'{oid} is already in this batch')
        if oid not in self.scene.objects:
            raise Rejected(f'unknown object {oid}')
        if len(self.brought_in) >= MAX_BROUGHT_IN:
            raise Rejected(f'at most {MAX_BROUGHT_IN} objects can be brought into a batch; already {self.brought_in}')
        st = self.state.objects[oid]
        home = next(b.index for b in self.plan.values() if oid in b.objects)
        later = home > self.batch.index
        self.members.append(oid)
        self.brought_in.append(oid)
        st.batch = self.batch.index
        if later:
            self.moved_out.add(oid)
            self._save_moved_out()
        if st.status in ('done', 'failed'):          # an earlier object reopened for fixing
            self.state.set_status(oid, 'working', note=f'brought into batch {self.batch.index}')
        # material: card + four-side view; and a frame that shows it
        self.observer.set_batch_colours(self.members)
        added_frame = None
        if not any(oid in self.observer.boxes([oid], s) for s in self.frames):
            added_frame = first_key_frame(self.scene.objects[oid])
            if added_frame not in self.frames:
                self.frames.append(added_frame)
                self.frames.sort()
        out = self.observer.out / 'opening' / f'brought_in_{oid}.png'
        self.observer.four_sides(oid).save(out)
        frames = {}
        for s in self.frames:
            im, _ = self.observer.boxed_frame(self.members, s)
            p = self.observer.out / 'opening' / f'key_frame_{s:04d}.png'
            im.save(p)
            frames[s] = p
        return self._note('bring_in', object=oid, home_batch=home, later=later, reason=reason,
                          added_frame=added_frame, card=self.observer.card(oid), four_sides=str(out),
                          key_frames={k: str(v) for k, v in frames.items()})

    # ------------------------------------------------------------ views

    def check_view(self, seq) -> int:
        if seq not in self.frames:
            raise Rejected(f'key frame {seq} is not one of this batch\'s frames {self.frames}')
        return seq

    # ------------------------------------------------------------ end of batch

    def end(self) -> dict:
        """Batch-end check: global contacts and a 15 s physics run. Objects with a problem are
        reopened once; a second time they are marked failed."""
        pending = self.remaining()
        if pending:
            raise Rejected(f'objects not finished yet: {pending}')
        v = self.verifier
        m = v.measure_now()
        problems = {}
        for pair in m.contacts:
            a, b = tuple(pair)
            if any(o in self.members for o in pair) and not _allowed(self.state, pair):
                for o in pair:
                    if o in self.members:
                        problems.setdefault(o, []).append({'check': 'penetration', 'with': (set(pair) - {o}).pop()})
        c = v.cfg
        for o in self.members:
            if m.wall_pen.get(o, 0) > c.wall_stop_m:
                problems.setdefault(o, []).append({'check': 'wall_pen', 'value': m.wall_pen[o]})
            if m.outside.get(o, 0) > c.room_stop_m:
                problems.setdefault(o, []).append({'check': 'outside', 'value': m.outside[o]})
        movable = set(self.members)
        for o in list(movable):
            movable.update(self.state.resting_on(o))
        phys = v.review_physics(seconds=END_PHYSICS_S, movable=sorted(movable)) if v.isaac else {'moved': [], 'undo': []}
        for item in phys['moved']:
            if item['level'] == 'stop' and item['object'] in self.members:
                problems.setdefault(item['object'], []).append({'check': 'physics', 'note': item['note']})
        # an object the agent gave up on stays failed; only finished ones are reopened
        problems = {o: v for o, v in problems.items() if self.state.objects[o].status == 'done'}
        reopened, failed = [], []
        for o, why in problems.items():
            if self.reopened.get(o, 0) < MAX_REOPEN:
                self.reopened[o] = self.reopened.get(o, 0) + 1
                self.state.set_status(o, 'working', note=f'batch end: {why[0]["check"]}')
                reopened.append(o)
            else:
                self.state.set_status(o, 'failed', note=f'batch end again: {why[0]["check"]}')
                failed.append(o)
        result = {'problems': problems, 'reopened': reopened, 'failed': failed,
                  'physics': {k: phys[k] for k in ('moved', 'undo') if k in phys}, 'ok': not reopened}
        self._note('end', **{k: result[k] for k in ('problems', 'reopened', 'failed', 'ok')})
        return result

    def close(self) -> dict:
        """Checkpoint: the version the next batch starts from."""
        if self.remaining():
            raise Rejected(f'objects not finished yet: {self.remaining()}')
        e = self._note('close', version=self.state.version, objects=self.members, brought_in=self.brought_in)
        cp = self.run / 'checkpoints.json'
        cps = json.loads(cp.read_text()) if cp.exists() else {}
        cps[str(self.batch.index)] = self.state.version
        cp.write_text(json.dumps(cps, indent=1))
        return e


def _allowed(state, pair) -> bool:
    a, b = tuple(pair)
    for o in (a, b):
        st = state.objects.get(o)
        if st and st.support.kind == 'object' and st.support.on in (a, b) and st.support.on != o:
            return True
    return False
