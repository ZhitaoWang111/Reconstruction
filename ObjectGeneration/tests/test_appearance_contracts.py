"""Appearance recovery/integrity contracts, without GPU or API calls."""
import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objgen.appearance.contracts import (atomic_json, digest, resolve_inputs,
                                         run_lock, valid_stage, validate_config)
from objgen.appearance.baseline import execute_stage


class AppearanceContractsTests(unittest.TestCase):
    def test_atomic_failure_preserves_old_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'report.json'
            atomic_json(p, {'status': 'success'})
            with self.assertRaises(ValueError):
                atomic_json(p, {'bad': float('nan')})
            self.assertEqual(json.loads(p.read_text()), {'status': 'success'})

    def test_corrupted_output_reruns_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = Path(tmp)
            report = {'stages': {}}
            calls = []
            def action():
                calls.append(1)
                (revision / 'result').write_text('verified')
            execute_stage(report, revision, 'baseline', action, ['result'])
            execute_stage(report, revision, 'baseline', action, ['result'])
            self.assertEqual(len(calls), 1)
            (revision / 'result').write_text('corrupted')
            execute_stage(report, revision, 'baseline', action, ['result'])
            self.assertEqual(len(calls), 2)
            self.assertTrue(valid_stage(report['stages']['baseline'], revision))

    def test_lock_rejects_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(tmp):
                with self.assertRaises(RuntimeError):
                    with run_lock(tmp):
                        pass
            with run_lock(tmp):
                pass

    def test_mask_manifest_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for p in ('input', 'sam3d/sam31', 'usd', 'physics'):
                (run / p).mkdir(parents=True)
            image, mask = run / 'input/rgb.png', run / 'sam3d/sam31/mask.png'
            for p in (image, mask, run / 'sam3d/mesh.glb', run / 'usd/object.usda'):
                p.write_bytes(b'snapshot')
            meta = {'image': str(image), 'mask': str(mask), 'mask_sha256': digest(mask)}
            atomic_json(run / 'input/input.json', meta)
            atomic_json(run / 'sam3d/report.json', {'status': 'success'})
            self.assertEqual(resolve_inputs(run)['mask']['path'], str(mask))
            mask.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                resolve_inputs(run)
            meta['mask'] = str(run.parent / 'outside.png')
            atomic_json(run / 'input/input.json', meta)
            with self.assertRaisesRegex(ValueError, 'outside run'):
                resolve_inputs(run)

    def test_failed_stage_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = Path(tmp)
            report = {'stages': {}}
            def action():
                (revision / 'partial').write_text('incomplete')
                raise RuntimeError('interrupted worker')
            with self.assertRaises(RuntimeError):
                execute_stage(report, revision, 'baseline', action, ['partial'])
            self.assertEqual(report['stages']['baseline']['status'], 'failed')
            self.assertFalse(valid_stage(report['stages']['baseline'], revision))


if __name__ == '__main__':
    unittest.main()
