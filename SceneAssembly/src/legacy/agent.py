"""Batch assembly agent: one model session places one batch of objects.

The model decides; the harness computes. Read-only tools return evidence
(object card, candidate sheet, compare view, turntable). Every placement goes
propose -> look -> commit: propose() returns the warnings and a render of the
move, and commit() needs a reason that answers them. Everything sent and
received is logged under the experiment directory.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import anthropic
import numpy as np

import handoff
from pose import Pose
from scene import Scene
from views import Views, jpeg

MODEL = 'claude-opus-5-5'
GATEWAY_KEY = Path('/home/seer/wzt/Gateway/key.json')

SYSTEM = """You assemble a 3-D scene from generated furniture assets so that it matches a scanned room.

Each object already has a generated USD asset (right shape and roughly right size, but its orientation is arbitrary) and a scan of where it stood. Your job is to give every object in the batch a pose: position, yaw, which way is up, and what it rests on. The harness does the geometry: it renders, measures, and checks. You judge: you look at the photos and decide what is right.

Coordinates: scan frame, metres, +Z up. A pose is (x, y, z, yaw_deg, up, scale_xy, scale_z). (x, y) is the centre of the object's footprint, z is the height of its lowest point (the floor is at floor_z; on top of another object it is that object's top surface), yaw_deg turns it about +Z counter-clockwise from +X, and up names which of the asset's own axes points to the sky (+Z for an asset that is already upright). scale_xy scales the footprint and scale_z the height, separately.

What the evidence can and cannot tell you:
- The scan covers only what the camera saw. Scan extents and registration residuals are evidence about position, not about which way an object faces: residuals of different yaws are usually within noise of each other.
- SAM3D estimated the orientation from one photo. Its up axis is usually right but sometimes upside down or tipped over; its yaw is often 180 deg off. Treat it as one candidate, not an answer.
- Which way an object faces is decided by looking: compare the candidate renders against the photo, and use the turntable to find the asset's front.
- Geometry finds contact, not direction: points of object A just under object B do not tell you which one rests on which.
- Generated assets usually have the right height but often the wrong proportions: a footprint 20-90% too large is common. The scanned top and bottom heights on the object card are the most reliable size evidence there is. Fix a wrong footprint with scale_xy and leave scale_z near 1 unless the heights say otherwise: shrinking an object uniformly also lowers its top, and everything resting on it sinks with it.
- Every object resting on another inherits that support's height errors. Check each support's top against the scanned bottom of what rests on it before building on it.

How to work:
- Place supporting objects before the things on them. Use get_object_card first; look at the candidate sheet before choosing an orientation.
- propose_pose never changes anything; it returns warnings and a render of the move. Look at the render, then commit_pose with a reason that answers every warning, or change the proposal. You may revise a placed object at any time; objects resting on it are marked stale and must be revisited.
- Use support="floor" or another object id; z is then filled in by lowering the object onto that support unless you give z explicitly.
- When an object is placed as well as the evidence allows, call accept_object. If an asset itself is wrong (wrong shape, far wrong size), say so with flag_object; flagging records the problem and does not stop you from placing the object as well as possible.
- Be economical: every image costs tokens. Do not re-request an image you already have unless something changed.
- Finish by calling finish_batch once every object is accepted or flagged."""


def _schema(props: dict, required: list[str]) -> dict:
    return {'type': 'object', 'properties': props, 'required': required, 'additionalProperties': False}


OID = {'type': 'string', 'description': 'object id, e.g. obj_0037'}
FRAME = {'type': 'string', 'description': "key frame role ('primary', 'secondary', 'third') or its seq number"}

TOOLS = [
    {'name': 'get_object_card', 'strict': True,
     'description': 'Text facts about one object: label, sizes (asset vs scan), physics, SAM3D orientation, the five candidate poses with residuals, contact points with other batch objects, and its key frames.',
     'input_schema': _schema({'oid': OID}, ['oid'])},
    {'name': 'view_candidates', 'strict': True,
     'description': 'One image: SAM3D and the four scan registrations rendered over the same key-frame photo, side by side, green = the object outline in the photo. Returns the silhouette IoU of each. Use it to decide orientation.',
     'input_schema': _schema({'oid': OID, 'frame': FRAME}, ['oid', 'frame'])},
    {'name': 'view_asset', 'strict': True,
     'description': "The asset alone seen from +X, +Y, -X, -Y with its own axes drawn, using the given up axis. Use it to find which side is the front.",
     'input_schema': _schema({'oid': OID, 'up': {'type': 'string', 'enum': ['+Z', '-Z', '+X', '-X', '+Y', '-Y']}}, ['oid', 'up'])},
    {'name': 'view_pose', 'strict': True,
     'description': 'Photo next to a render of the object at its current committed pose (and every other placed object in grey) under a key-frame camera.',
     'input_schema': _schema({'oid': OID, 'frame': FRAME}, ['oid', 'frame'])},
    {'name': 'fit_to_scan', 'strict': True,
     'description': 'Bounded scan registration starting from a pose you give: yaw may move at most yaw_window_deg, XY at most xy_window_m; z, up and both scales stay. Returns the fitted pose and residuals. Does not change the scene.',
     'input_schema': _schema({'oid': OID, 'x': {'type': 'number'}, 'y': {'type': 'number'}, 'yaw_deg': {'type': 'number'},
                              'up': {'type': 'string', 'enum': ['+Z', '-Z', '+X', '-X', '+Y', '-Y']},
                              'scale_xy': {'type': 'number'}, 'scale_z': {'type': 'number'},
                              'yaw_window_deg': {'type': 'number'}, 'xy_window_m': {'type': 'number'}},
                             ['oid', 'x', 'y', 'yaw_deg', 'up', 'scale_xy', 'scale_z', 'yaw_window_deg', 'xy_window_m'])},
    {'name': 'propose_pose', 'strict': True,
     'description': 'Check a pose without applying it. Returns warnings (penetration, gap to support, hover, height vs scan, scale, outside scan) and a photo-vs-render image of the proposal. z: give a number, or null to rest it on its support.',
     'input_schema': _schema({'oid': OID, 'x': {'type': 'number'}, 'y': {'type': 'number'},
                              'z': {'type': ['number', 'null']}, 'yaw_deg': {'type': 'number'},
                              'up': {'type': 'string', 'enum': ['+Z', '-Z', '+X', '-X', '+Y', '-Y']},
                              'scale_xy': {'type': 'number', 'description': 'footprint scale, 1 = generated size'},
                              'scale_z': {'type': 'number', 'description': 'height scale, 1 = generated size'},
                              'support': {'type': ['string', 'null'], 'description': "'floor', an object id, or null if it rests on nothing (hangs)"},
                              'frame': FRAME, 'reason': {'type': 'string'}},
                             ['oid', 'x', 'y', 'z', 'yaw_deg', 'up', 'scale_xy', 'scale_z', 'support', 'frame', 'reason'])},
    {'name': 'commit_pose', 'strict': True,
     'description': 'Apply a proposal returned by propose_pose. The reason must answer each of its warnings. Objects resting on the moved object become stale.',
     'input_schema': _schema({'proposal_id': {'type': 'string'}, 'reason': {'type': 'string'}}, ['proposal_id', 'reason'])},
    {'name': 'accept_object', 'strict': True,
     'description': 'Mark a placed object as done.',
     'input_schema': _schema({'oid': OID, 'note': {'type': 'string'}}, ['oid', 'note'])},
    {'name': 'flag_object', 'strict': True,
     'description': 'Record that the asset itself is wrong (shape, size) so a human can regenerate it.',
     'input_schema': _schema({'oid': OID, 'problem': {'type': 'string'}}, ['oid', 'problem'])},
    {'name': 'finish_batch', 'strict': True,
     'description': 'End the session once every object is accepted or flagged.',
     'input_schema': _schema({'summary': {'type': 'string'}}, ['summary'])},
]


def _image_block(im) -> dict:
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                        'data': base64.standard_b64encode(jpeg(im)).decode()}}


def _positive(*scales):
    if not all(0.2 <= s <= 5 for s in scales):
        raise ValueError('scales must be between 0.2 and 5')


class Harness:
    def __init__(self, scene: Scene, out: Path):
        self.s = scene
        self.v = Views(scene)
        self.out = Path(out)
        (self.out / 'images').mkdir(parents=True, exist_ok=True)
        self.finished = None
        self._img = 0

    def _save(self, im, tag) -> str:
        self._img += 1
        name = f'images/{self._img:03d}_{tag}.jpg'
        im.save(self.out / name, quality=85)
        return name

    def _oid(self, oid):
        if oid not in self.s.assets:
            raise ValueError(f'{oid} is not in this batch; batch is {", ".join(self.s.oids)}')
        return oid

    # each returns (content blocks, short log line)
    def get_object_card(self, oid):
        oid = self._oid(oid)
        a = self.s.assets[oid]
        ob = a.ob
        slo, shi = a.scan.min(0), a.scan.max(0)
        zlo, zhi = np.percentile(a.scan[:, 2], [2, 98])
        lines = [f'{oid} · label {ob.label}' + (' (annotator could not name it)' if ob.generic else ''),
                 f'asset size (own frame, x y z): {ob.dims[0]:.2f} x {ob.dims[1]:.2f} x {ob.dims[2]:.2f} m; '
                 f'scale source: {ob.size_source} (x{ob.scale_factor:.3f})',
                 f'scan extent (what was seen; partial scans are smaller than the object): {shi[0]-slo[0]:.2f} x {shi[1]-slo[1]:.2f} m footprint, '
                 f'centre ({(slo[0]+shi[0])/2:.3f}, {(slo[1]+shi[1])/2:.3f})',
                 f'scanned heights (2nd-98th percentile, robust to stray points): bottom z {zlo:.3f}, top z {zhi:.3f}, '
                 f'so about {zhi-zlo:.2f} m tall where seen',
                 f'physics: {ob.body_type}, {ob.collider} collider',
                 'candidates (residual = median distance of asset surface to scan; cannot tell front from back):']
        for c in self.s.candidates(oid):
            p = c['pose']
            extra = f", SAM3D tilt from that up axis {c['sam3d_tilt_deg']} deg" if c['name'] == 'sam3d' else ''
            lines.append(f"  {c['name']}: x {p.x:.3f} y {p.y:.3f} z {p.z:.3f} yaw {p.yaw_deg:.1f} up {p.up} "
                         f"· residual {c['residual_cm']} cm · inliers(<5cm) {c['inliers']:.0%}"
                         + (' · fit made it worse, kept start' if c['diverged'] else '') + extra)
        cs = self.s.contacts(oid)
        lines.append('scan contact just below its lowest scan points (direction unknown): '
                     + ('; '.join(f"{d['oid']} {d['points']} pts at z {d['z'][0]}-{d['z'][1]}" for d in cs) if cs else 'none'))
        lines.append('key frames: ' + ', '.join(f'{f.role} seq {f.seq} ({f.visible_pixels/ (self.s.handoff.camera.width*self.s.handoff.camera.height):.1%} of frame)'
                                                for f in ob.key_frames))
        p = self.s.placed.get(oid)
        lines.append('current: ' + (f'{p.status}, ' + json.dumps(p.pose.to_dict()) + f', support {p.support}' if p else 'unplaced'))
        return [{'type': 'text', 'text': '\n'.join(lines)}], f'card {oid}'

    def view_candidates(self, oid, frame):
        oid = self._oid(oid)
        im, ious, seq = self.v.candidate_sheet(oid, frame)
        name = self._save(im, f'cand_{oid}_{seq}')
        names = [c['name'] for c in self.s.candidates(oid)]
        txt = (f'{oid} candidates under key frame seq {seq}, left to right: ' + ', '.join(names)
               + '. Silhouette IoU with the photo mask: ' + ', '.join(f'{n} {i:.2f}' for n, i in zip(names, ious))
               + '. IoU differences under ~0.05 are not meaningful; judge by shape and features.')
        return [_image_block(im), {'type': 'text', 'text': txt}], f'candidates {oid} seq {seq} -> {name}'

    def view_asset(self, oid, up):
        oid = self._oid(oid)
        im = self.v.turntable(oid, up)
        name = self._save(im, f'asset_{oid}_{up}')
        return [_image_block(im), {'type': 'text', 'text': f'{oid} with asset axis {up} up. Axes: red X, green Y, blue Z (asset frame after the up turn, before yaw).'}], f'asset {oid} {up} -> {name}'

    def view_pose(self, oid, frame):
        oid = self._oid(oid)
        p = self.s.placed.get(oid)
        if p is None:
            raise ValueError(f'{oid} has no committed pose yet; use propose_pose to see a pose')
        im, iou, seq = self.v.compare(oid, p.pose, frame)
        name = self._save(im, f'pose_{oid}_{seq}')
        return [_image_block(im), {'type': 'text', 'text': f'{oid} committed pose under seq {seq}; silhouette IoU {iou:.2f}.'}], f'view {oid} seq {seq} -> {name}'

    def fit_to_scan(self, oid, x, y, yaw_deg, up, scale_xy, scale_z, yaw_window_deg, xy_window_m):
        oid = self._oid(oid)
        _positive(scale_xy, scale_z)
        p = self.s.placed.get(oid)
        z = p.pose.z if p else float(self.s.assets[oid].scan.min(0)[2])
        fitted, info = self.s.fit(oid, Pose(x, y, z, yaw_deg, up, scale_xy, scale_z),
                                  min(abs(yaw_window_deg), 45), min(abs(xy_window_m), 0.5))
        txt = (f'fitted: x {fitted.x:.3f} y {fitted.y:.3f} yaw {fitted.yaw_deg:.1f} up {fitted.up} '
               f'(residual {info["start_residual_cm"]} -> {info["residual_cm"]} cm'
               + (', fit could not improve, start kept' if info['kept_start'] else '') + '). Nothing changed; propose it to use it.')
        return [{'type': 'text', 'text': txt}], f'fit {oid}'

    def propose_pose(self, oid, x, y, z, yaw_deg, up, scale_xy, scale_z, support, frame, reason):
        oid = self._oid(oid)
        if support is not None and support != 'floor' and support not in self.s.assets:
            raise ValueError(f"support must be 'floor', null, or an object in this batch")
        _positive(scale_xy, scale_z)
        pose = Pose(float(x), float(y), 0.0, float(yaw_deg) % 360, up, float(scale_xy), float(scale_z))
        if z is None:
            if support is None:
                raise ValueError('z is null but no support given; give z or a support')
            rest = self.s.rest_z(oid, pose, support)
            if rest is None:
                raise ValueError(f'lowered straight down from ({x:.2f}, {y:.2f}) it does not land on {support}; '
                                 'check x, y or place the support first')
            pose = pose.with_z(rest)
        else:
            pose = pose.with_z(float(z))
        prop = self.s.propose(oid, pose, support, reason)
        im, iou, seq = self.v.compare(oid, pose, frame)
        name = self._save(im, f'prop_{oid}_{prop.pid}')
        w = '\n'.join(f'- {x}' for x in prop.warnings) or '- none'
        txt = (f'proposal {prop.pid} for {oid}: ' + json.dumps(pose.to_dict()) + f', support {support}\n'
               f'warnings:\n{w}\nsilhouette IoU under seq {seq}: {iou:.2f}. Nothing has changed yet; commit_pose {prop.pid} to apply.')
        return [_image_block(im), {'type': 'text', 'text': txt}], f'propose {oid} {prop.pid} warnings={len(prop.warnings)} -> {name}'

    def commit_pose(self, proposal_id, reason):
        p = self.s.proposals.get(proposal_id)
        stale = self.s.commit(proposal_id, reason)
        txt = f'committed {proposal_id} ({p.oid}) as version {self.s.version}.'
        if stale:
            txt += ' Now stale, revisit: ' + ', '.join(stale)
        return [{'type': 'text', 'text': txt + '\nState:\n' + self.s.summary()}], f'commit {proposal_id} {p.oid}'

    def accept_object(self, oid, note):
        oid = self._oid(oid)
        self.s.accept(oid, note)
        return [{'type': 'text', 'text': f'{oid} accepted.\nState:\n' + self.s.summary()}], f'accept {oid}'

    def flag_object(self, oid, problem):
        oid = self._oid(oid)
        self.s.flag(oid, problem)
        return [{'type': 'text', 'text': f'{oid} flagged.'}], f'flag {oid}'

    def finish_batch(self, summary):
        open_ = [o for o in self.s.oids if not ((p := self.s.placed.get(o)) and p.status == 'accepted') and o not in self.s.flags]
        if open_:
            raise ValueError('not finished: ' + ', '.join(open_) + ' are neither accepted nor flagged')
        self.finished = summary
        return [{'type': 'text', 'text': 'batch finished.'}], 'finish'

    def run(self, name, args):
        try:
            content, log = getattr(self, name)(**args)
            return content, log, False
        except (ValueError, KeyError, TypeError) as e:
            return [{'type': 'text', 'text': f'error: {e}'}], f'{name} error: {e}', True


def client() -> anthropic.Anthropic:
    k = json.loads(GATEWAY_KEY.read_text())
    return anthropic.Anthropic(api_key=k['token'], base_url=k['Claude_URL'].rstrip('/'), max_retries=4, timeout=600)


def run_batch(scene: Scene, out: Path, effort='high', max_turns=60, budget_tokens=1_500_000):
    out = Path(out)
    h = Harness(scene, out)
    api = client()
    hs = scene.handoff
    first = (f'Scene {hs.scene_id}. floor_z = {hs.floor_z:.4f} m. Camera {hs.camera.width}x{hs.camera.height}.\n'
             f'This batch: ' + ', '.join(f'{o} ({scene.assets[o].ob.label})' for o in scene.oids) + '.\n'
             'Every object starts unplaced. Place, check, and accept or flag each one, then call finish_batch.')
    messages = [{'role': 'user', 'content': first}]
    log = open(out / 'calls.jsonl', 'a')
    used = {'input': 0, 'output': 0}
    t_start = time.time()
    for turn in range(1, max_turns + 1):
        t0 = time.time()
        with api.beta.messages.stream(model=MODEL, max_tokens=32000, system=SYSTEM, tools=TOOLS, messages=messages,
                                      thinking={'type': 'adaptive', 'display': 'summarized'},
                                      output_config={'effort': effort},
                                      betas=['server-side-fallback-2026-07-01'], fallbacks='default') as stream:
            r = stream.get_final_message()
        used['input'] += r.usage.input_tokens
        used['output'] += r.usage.output_tokens
        messages.append({'role': 'assistant', 'content': r.content})
        record = {'turn': turn, 'seconds': round(time.time() - t0, 1), 'stop': r.stop_reason, 'model': r.model,
                  'usage': {'input': r.usage.input_tokens, 'output': r.usage.output_tokens},
                  'thinking': [b.thinking for b in r.content if b.type == 'thinking' and b.thinking],
                  'text': [b.text for b in r.content if b.type == 'text'],
                  'calls': []}
        if r.stop_reason == 'refusal':
            record['refusal'] = str(getattr(r, 'stop_details', None))
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
            raise RuntimeError(f'model refused at turn {turn}: {record["refusal"]}')
        calls = [b for b in r.content if b.type == 'tool_use']
        if not calls:
            record['note'] = 'no tool call; nudged'
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
            log.flush()
            messages.append({'role': 'user', 'content': 'Continue with the tools. Call finish_batch when every object is accepted or flagged.'})
            continue
        results = []
        for b in calls:
            content, line, err = h.run(b.name, dict(b.input))
            results.append({'type': 'tool_result', 'tool_use_id': b.id, 'content': content, **({'is_error': True} if err else {})})
            record['calls'].append({'tool': b.name, 'input': b.input, 'result': line, 'error': err})
            print(f'[{turn:02d}] {line}', flush=True)
        messages.append({'role': 'user', 'content': results})
        log.write(json.dumps(record, ensure_ascii=False) + '\n')
        log.flush()
        if h.finished is not None:
            break
        if used['input'] > budget_tokens:
            print(f'stopping: input token budget {budget_tokens} exceeded', flush=True)
            break
    summary = {'turns': turn, 'finished': h.finished is not None, 'summary': h.finished, 'tokens': used,
               'seconds': round(time.time() - t_start), 'version': scene.version,
               'state': {o: p.to_dict() for o, p in scene.placed.items()}, 'flags': scene.flags}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary
