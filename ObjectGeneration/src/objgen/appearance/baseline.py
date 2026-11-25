"""M1 baseline entry: captured inputs, immutable revisions and hash-based resume."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import yaml

from .contracts import (allocate_revision, atomic_json, digest, json_digest,
                        resolve_inputs, run_lock, valid_stage, validate_config)

from ..config import ROOT


def prepare_images(inputs, revision):
    with Image.open(inputs['image']['path']) as source, Image.open(inputs['mask']['path']) as raw_mask:
        image = source.convert('RGB')
        mask = raw_mask.convert('L').point(lambda v: 255 if v > 0 else 0)
        if image.size != mask.size:
            raise ValueError('RGB/mask dimensions differ')
        box = mask.getbbox()
        if box is None:
            raise ValueError('Empty segmentation mask')
        image.crop(box).save(revision / 'reference_crop.png')
        mask.crop(box).save(revision / 'reference_mask.png')
        background = Image.new('RGB', image.size, (128, 128, 128))
        Image.composite(image, background, mask).crop(box).save(revision / 'reference_isolated.png')
        atomic_json(revision / 'reference.json', {'image_size': list(image.size), 'bbox_xyxy': list(box)})


def execute_stage(report, revision, name, action, outputs):
    old = report['stages'].get(name, {})
    if valid_stage(old, revision):
        print('APPEARANCE_STAGE_REUSED', name, flush=True)
        return
    report['status'] = name
    report['stages'][name] = {'status': 'running', 'attempt': old.get('attempt', 0) + 1}
    atomic_json(revision / 'report.json', report)
    try:
        action()
        hashes = {p: digest(revision / p) for p in outputs}
        report['stages'][name].update(status='success', outputs=hashes)
    except BaseException as exc:
        report['stages'][name].update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                     error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        atomic_json(revision / 'report.json', report)


def build(run, config, resume=None, stop_after=None):
    with run_lock(run):
        inputs = resolve_inputs(run)
        source_dir = Path(__file__).parent
        tools = {p.name: digest(p) for p in sorted(source_dir.glob('*.py'))}
        tools['blender_binary'] = digest(config['blender'])
        fingerprint = json_digest({'inputs': inputs, 'config': config, 'tools': tools})
        if resume:
            revision = (run / 'usd/appearance' / resume).resolve()
            if revision.parent != (run / 'usd/appearance').resolve() or not resume.startswith('v_'):
                raise ValueError('Invalid revision ID')
            manifest = json.loads((revision / 'manifest.json').read_text())
            if manifest['fingerprint'] != fingerprint:
                raise ValueError('Inputs/config/tools changed; create a new revision')
            report = json.loads((revision / 'report.json').read_text())
        else:
            revision = allocate_revision(run)
            (revision / 'renders').mkdir()
            manifest = {'schema_version': 1, 'revision': revision.name, 'run_dir': str(run),
                        'created_at': datetime.now().astimezone().isoformat(), 'fingerprint': fingerprint,
                        'inputs': inputs, 'tool_sha256': tools}
            atomic_json(revision / 'manifest.json', manifest)
            atomic_json(revision / 'config.json', config)
            report = {'schema_version': 1, 'milestone': 'M1', 'revision': revision.name,
                      'status': 'pending', 'passed': False, 'stages': {}}
            atomic_json(revision / 'report.json', report)
        print('APPEARANCE_REVISION', revision, flush=True)

        def geometry():
            env = dict(os.environ, OPENBLAS_CORETYPE='Haswell', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            with (revision / 'geometry.log').open('w') as log:
                subprocess.run(['taskset', '-c', '0', str(ROOT / '.venv-sam3d/bin/python'),
                                str(source_dir / 'extract_geometry.py'), inputs['usd']['path'], str(revision)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)

        def render():
            with (revision / 'blender.log').open('w') as log:
                subprocess.run([config['blender'], '--background', '--factory-startup', '--python-exit-code', '1',
                                '--python', str(source_dir / 'baseline_blender.py'), '--', str(revision)],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            if 'APPEARANCE_BASELINE_RENDERED' not in (revision / 'blender.log').read_text():
                raise RuntimeError('Missing baseline render marker')
            for name in views:
                with Image.open(revision / 'renders' / (name + '.png')) as image:
                    assert image.size == (512, 512)
                    image.verify()

        views = ('front', 'back', 'left', 'right', 'top', 'three_quarter')
        stages = [
            ('geometry', geometry, ['canonical_mesh.npz', 'geometry.json']),
            ('reference', lambda: prepare_images(inputs, revision),
             ['reference_crop.png', 'reference_mask.png', 'reference_isolated.png', 'reference.json']),
            ('render', render, ['work.blend', 'baseline_cameras.json', *['renders/' + v + '.png' for v in views]])]
        # If an upstream artifact is damaged, invalidate downstream work as well.
        invalid = False
        for name, _, _ in stages:
            invalid = invalid or not valid_stage(report['stages'].get(name, {}), revision)
            if invalid and name in report['stages']:
                report['stages'][name]['status'] = 'invalidated'
        report.pop('error', None)
        report['passed'] = False
        try:
            for name, action, outputs in stages:
                execute_stage(report, revision, name, action, outputs)
                if stop_after == name:
                    report['status'] = 'paused'
                    atomic_json(revision / 'report.json', report)
                    return revision
            if resolve_inputs(run) != inputs:
                raise RuntimeError('Source inputs changed while producing baseline')
            report.update(status='baseline_ready', passed=True)
        except BaseException as exc:
            report.update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=f'{type(exc).__name__}: {exc}', passed=False)
            raise
        finally:
            atomic_json(revision / 'report.json', report)
        return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/appearance/default.yaml')
    parser.add_argument('--resume', help='Existing revision ID (v_<timestamp>); fingerprint must match')
    parser.add_argument('--stop-after', choices=['geometry', 'reference'], help='Controlled checkpoint for recovery verification')
    args = parser.parse_args()
    config = validate_config(yaml.safe_load(args.config.read_text()))
    try:
        revision = build(args.run_dir.resolve(), config, args.resume, args.stop_after)
    except Exception as exc:
        print(f'APPEARANCE_BASELINE_FAILED {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print('APPEARANCE_BASELINE_REPORT', revision / 'report.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
