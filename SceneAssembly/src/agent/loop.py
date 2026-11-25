"""The conversation loop for one batch, and the tool handlers.

Every tool answers with text (a compact JSON) and, where it makes one, a stitched image. Images go
inside the function_call_output (the gateway accepts it). The history is only appended to.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from batch import BatchSession
from observe.plan import plan_view
from state import Pose, Rejected, Support

from .llm import Client, image_part, text_part
from .tools import INSTRUCTIONS, TOOLS

MAX_TURNS = 80
MAX_INPUT_TOKENS = 3_000_000
MAX_IDLE = 2


@dataclass
class Context:
    scene: object
    structure: object
    state: object
    initial: object
    observer: object
    verifier: object
    isaac: object
    plan: list


def _round(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    return x


class Tools:
    """Tool handlers: each returns {'text': str, 'image': path or None, 'end': bool}."""

    def __init__(self, ctx: Context, session: BatchSession):
        self.ctx, self.s = ctx, session
        self.ended = False

    # ------------------------------------------------------------ helpers

    def _pose(self, a) -> tuple:
        sup = a['support']
        against = tuple(sup['against']) if sup.get('against') else ()
        kind = sup['kind']
        support = Support(kind=kind, on=sup.get('on') if kind == 'object' else None,
                          surface_z=sup.get('surface_z') if kind == 'object' else None,
                          height=sup.get('height') if kind == 'wall' else None, against=against)
        pose = Pose(x=a['x'], y=a['y'], yaw_deg=a['yaw_deg'], up=a['up'], scale=tuple(a['scale']),
                    tilt_deg=a['tilt_deg'], tilt_toward_deg=a['tilt_toward_deg'])
        return pose, support

    def _member(self, oid):
        if oid not in self.s.members:
            raise Rejected(f'{oid} is not in this batch (members: {self.s.members}); bring_in it first')

    def _working(self, oid):
        self._member(oid)
        if self.ctx.state.objects[oid].status == 'untouched':
            raise Rejected(f'begin {oid} first')

    def _image(self, oid, seq=None):
        img, path = (self.ctx.observer.stitched(oid, seq) if seq is not None else self.ctx.observer.after_change(oid))
        return str(path)

    def _issues_text(self, issues):
        out = []
        for i in issues:
            if i['level'] == 'ok':
                continue
            item = {'check': i['check'], 'level': i['level'], 'value': _round(i.get('value')), 'before': _round(i.get('before'))}
            if i.get('with'):
                item['with'] = i['with']
            if i.get('note'):
                item['note'] = i['note']
            out.append(item)
        return out

    # ------------------------------------------------------------ tools

    def set_order(self, a):
        e = self.s.set_order(a['order'], a['reason'])
        return {'text': json.dumps({'ok': True, 'order': e['order']})}

    def begin(self, a):
        oid = a['object']
        self.s.start(oid)
        st = self.ctx.state.objects[oid]
        return {'text': json.dumps({'ok': True, 'object': oid, 'pose': st.pose.to_dict(), 'support': st.support.to_dict(),
                                    'bottom_z': round(st.bottom_z, 3), 'image': f'key frame {self.ctx.observer.first_key_frame(oid)}'}),
                'image': self._image(oid)}

    def set_pose(self, a):
        oid = a['object']
        self._working(oid)
        pose, support = self._pose(a)
        before = self.ctx.verifier.baseline()
        try:
            change = self.ctx.state.set_pose(oid, pose, support, a['reason'])
        except Rejected as e:
            return {'text': json.dumps({'ok': False, 'rejected': e.reason})}
        image = self._image(oid)                      # the pose as set: kept or refused, one render either way
        r = self.ctx.verifier.review(oid, before, change, reason=a['reason'])
        if r['verdict'] == 'undo':
            image = _mark_refused(image)
        moved = [{'object': m['object'], 'because': m.get('because', 'this change'), 'bottom_z': m.get('bottom_z'),
                  **({'walls': m['walls']} if m.get('walls') else {})} for m in change.get('moved', [])]
        st = self.ctx.state.objects[oid]
        out = {'verdict': r['verdict'], 'issues': self._issues_text(r['issues']), 'moved': moved,
               'now': {'pose': st.pose.to_dict(), 'support': st.support.to_dict(), 'bottom_z': round(st.bottom_z, 3),
                       'status': st.status}}
        if r['verdict'] == 'undo':
            out['undo_streak'] = r.get('streak')
            if r.get('failed'):
                out['failed'] = 'three automatic undos in a row: the object is marked failed'
        return {'text': json.dumps(out, ensure_ascii=False), 'image': image}

    def undo(self, a):
        oid = a['object']
        self._working(oid)
        rep = self.ctx.state.undo(oid, a['reason'])
        return {'text': json.dumps({'ok': True, 'back_to_version': rep.get('back_to_version'),
                                    'moved': [m['object'] for m in rep['moved']]}), 'image': self._image(oid)}

    def view(self, a):
        oid = a['object']
        self._working(oid)
        seq = self.s.check_view(a['frame'])
        return {'text': json.dumps({'ok': True, 'object': oid, 'frame': seq}), 'image': self._image(oid, seq)}

    def check_scene(self, a):
        m = self.ctx.verifier.measure_now()
        cfg = self.ctx.verifier.cfg
        from verify.check import _allowed
        contacts = [sorted(p) for p in m.contacts if not _allowed(self.ctx.state, p)]
        out = {'contacts_not_declared_as_support': contacts,
               'in_wall_m': {o: v for o, v in m.wall_pen.items() if v > cfg.wall_report_m},
               'outside_room_m': {o: v for o, v in m.outside.items() if v > cfg.room_report_m},
               'support_gap_m': {o: v for o, v in m.support_gap.items()
                                 if v is not None and (v > cfg.support_float_report_m or v < -cfg.support_sink_report_m)},
               'wall_gap_m': {o: v for o, v in m.wall_gap.items() if v > cfg.wall_gap_report_m}}
        return {'text': json.dumps(_round(out))}

    def finish(self, a):
        oid = a['object']
        self._working(oid)
        movable = [oid, *self.ctx.state.resting_on(oid)]
        ph = self.ctx.verifier.review_physics(seconds=5.0, movable=movable)
        moved = [{k: i[k] for k in ('object', 'dz', 'dxy', 'tilt_deg', 'settled', 'level', 'note')} for i in ph['moved']]
        if ph['undo']:
            return {'text': json.dumps({'ok': False, 'physics': moved, 'undone': ph['undo'],
                                        'note': 'fell over or dropped: last change undone, the object stays open'}),
                    'image': self._image(oid)}
        self.s.finish(oid, a['note'])
        return {'text': json.dumps({'ok': True, 'physics': moved, 'fixed': ph.get('fixed', []),
                                    'remaining': self.s.remaining()})}

    def give_up(self, a):
        oid = a['object']
        self._member(oid)
        self.ctx.state.set_status(oid, 'failed', note=a['problem'])
        return {'text': json.dumps({'ok': True, 'remaining': self.s.remaining()})}

    def bring_in(self, a):
        e = self.s.bring_in(a['object'], a['reason'])
        oid = a['object']
        parts = [f"{oid} is now in this batch.\n" + self.ctx.observer.card_text(oid)]
        imgs = [e['four_sides']] + [e['key_frames'][s] for s in sorted(e['key_frames']) if e.get('added_frame') == s]
        return {'text': json.dumps({'ok': True, 'object': oid, 'frames': self.s.frames, 'added_frame': e.get('added_frame'),
                                    'card': parts[0]}, ensure_ascii=False),
                'images': imgs}

    def end_batch(self, a):
        r = self.s.end()
        if r['reopened']:
            return {'text': json.dumps({'ok': False, 'reopened': r['reopened'],
                                        'problems': {o: [p['check'] for p in v] for o, v in r['problems'].items()},
                                        'note': 'these objects are open again; fix them and finish them, then end_batch'})}
        self.ended = True
        failed = [o for o in self.s.members if self.ctx.state.objects[o].status == 'failed']
        return {'text': json.dumps({'ok': True, 'failed': failed, 'summary': a['summary']}), 'end': True}

    def run(self, name, args):
        fn = getattr(self, name, None)
        if fn is None or name.startswith('_') or name == 'run':
            return {'text': json.dumps({'ok': False, 'error': f'unknown tool {name}'})}
        try:
            return fn(args)
        except Rejected as e:
            return {'text': json.dumps({'ok': False, 'rejected': e.reason}, ensure_ascii=False)}


def _mark_refused(path) -> str:
    """Stamp a stitched image as the refused (undone) pose."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(path).convert('RGB')
    d = ImageDraw.Draw(im)
    w = im.size[0]
    d.rectangle([w // 2 + 3, 30, w, 66], fill=(200, 30, 30))
    d.text((w // 2 + 12, 34), 'REFUSED - this pose was undone', font=ImageFont.truetype(
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 22), fill=(255, 255, 255))
    out = str(path).replace('.png', '_refused.png')
    im.save(out)
    return out


# ---------------------------------------------------------------- the opening message

def opening(ctx: Context, session: BatchSession, pack: dict, out: Path) -> list:
    obs = ctx.observer
    cams = {}
    fov = math.degrees(2 * math.atan(ctx.scene.camera.width / 2 / ctx.scene.camera.fx))
    for s in session.frames:
        cams[s] = (obs.frame(s).c2w, fov)
    plan = plan_view(ctx.structure, cams)
    plan_path = out / 'plan_view.png'
    plan.save(plan_path)
    walls = [{'id': w.id, 'length_m': round(w.length, 2), 'role': w.role,
              'openings': [[round(o.from_m, 2), round(o.to_m, 2)] for o in w.openings]}
             for w in ctx.structure.walls.values()]
    placed = pack['placed']
    text = [f"Batch {pack['batch']}: objects {session.members}; shared key frames {session.frames}.",
            '', 'Object cards:', pack['cards_text'], '',
            'Already placed by earlier batches (rendered when in view; not yours to move unless you bring one in):',
            json.dumps(placed, ensure_ascii=False) if placed else '(none yet)', '',
            'Walls (ids used in the plan view and in `against`):', json.dumps(walls), '',
            'Images follow: the plan view, the boxed key frames, then one four-side view per object.']
    parts = [text_part('\n'.join(text)), text_part('Plan view:'), image_part(plan_path)]
    for s, p in sorted(pack['key_frames'].items()):
        parts += [text_part(f'Key frame {s}:'), image_part(p)]
    for o, p in pack['four_sides'].items():
        parts += [text_part(f'Four-side view of {o}:'), image_part(p)]
    parts.append(text_part('Start with set_order.'))
    return [{'role': 'user', 'content': parts}]


# ---------------------------------------------------------------- the loop

def run_batch(index: int, ctx: Context, out_dir: Path, session: BatchSession | None = None, effort='high',
              max_turns=MAX_TURNS, max_input_tokens=MAX_INPUT_TOKENS) -> dict:
    out = Path(out_dir) / f'batch_{index:02d}'
    out.mkdir(parents=True, exist_ok=True)
    s = session or BatchSession(ctx.plan, ctx.scene, ctx.structure, ctx.state, ctx.initial, ctx.observer, ctx.verifier, out_dir)
    pack = s.open(index)
    tools = Tools(ctx, s)
    client = Client(out / 'llm', effort=effort)
    instructions = INSTRUCTIONS.format(floor_z=ctx.structure.floor_z, storey=ctx.structure.storey_height)
    items = opening(ctx, s, pack, out)
    calls_log = open(out / 'calls.jsonl', 'a')
    used_in, used_out, usd, idle, turn = 0, 0, 0.0, 0, 0
    t0 = time.time()
    stop = 'max_turns'
    for turn in range(1, max_turns + 1):
        r = client.respond(instructions, items, TOOLS)
        used_in += r['usage'].get('input_tokens', 0) or 0
        used_out += r['usage'].get('output_tokens', 0) or 0
        usd += r['cost']
        items.extend(r['output'])
        rec = {'turn': turn, 'seconds': r['seconds'], 'input_tokens': r['usage'].get('input_tokens'),
               'output_tokens': r['usage'].get('output_tokens'), 'usd': round(r['cost'], 4), 'text': r['text'],
               'reasoning': r['reasoning'], 'calls': []}
        if not r['calls']:
            idle += 1
            if idle > MAX_IDLE:
                stop = 'idle'
                calls_log.write(json.dumps(rec, ensure_ascii=False) + '\n')
                break
            items.append({'role': 'user', 'content': [text_part(
                'Use a tool. When every object is done or failed, call end_batch.')]})
            calls_log.write(json.dumps(rec, ensure_ascii=False) + '\n')
            continue
        idle = 0
        for c in r['calls']:
            try:
                args = json.loads(c['arguments'])
            except json.JSONDecodeError as e:
                res = {'text': json.dumps({'ok': False, 'error': f'arguments are not valid JSON: {e}'})}
            else:
                res = tools.run(c['name'], args)
            content = [text_part(res['text'])]
            imgs = res.get('images') or ([res['image']] if res.get('image') else [])
            for p in imgs:
                content.append(image_part(p))
            items.append({'type': 'function_call_output', 'call_id': c['call_id'], 'output': content})
            rec['calls'].append({'name': c['name'], 'arguments': c['arguments'], 'result': res['text'][:2000],
                                 'images': [str(p) for p in imgs]})
        calls_log.write(json.dumps(rec, ensure_ascii=False) + '\n')
        calls_log.flush()
        if tools.ended:
            stop = 'end_batch'
            break
        if used_in > max_input_tokens:
            stop = 'token_budget'
            break
    calls_log.close()
    # out of turns or budget: whatever is still open is marked failed at its last pose
    leftover = s.remaining()
    if stop != 'end_batch':
        for o in leftover:
            ctx.state.set_status(o, 'failed', note=f'batch stopped: {stop}')
    cp = s.close()
    summary = {'batch': index, 'stop': stop, 'turns': turn, 'seconds': round(time.time() - t0, 1),
               'input_tokens': used_in, 'output_tokens': used_out, 'usd': round(usd, 2),
               'objects': {o: ctx.state.objects[o].status for o in s.members}, 'failed_by_stop': leftover if stop != 'end_batch' else [],
               'checkpoint_version': cp['version']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary
