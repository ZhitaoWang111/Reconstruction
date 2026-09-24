"""Publish a complete GPT viewer at the existing URL while retaining baseline assets."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

FRAME_PATHS = ('asset', 'npz', 'ply', 'rgb_image', 'original_rgb_image',
               'annotated_image', 'predictions_json')


def rebase_manifest(manifest, source, destination):
    result = copy.deepcopy(manifest)
    def relative(value):
        path = (source / value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return os.path.relpath(path, destination)
    result['background_asset'] = relative(result['background_asset'])
    for frame in result['frames']:
        for key in FRAME_PATHS:
            if key in frame:
                frame[key] = relative(frame[key])
        for item in frame['instances']:
            item['mask_file'] = relative(item['mask_file'])
    return result


def write_manifest(folder, manifest):
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (folder / 'manifest.js').write_text('window.MASK_PROJECTION_MANIFEST=' +
        json.dumps(manifest, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+';\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination:
        raise ValueError('Build into a separate directory before publishing')
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest.get('method') != 'gpt' or manifest.get('model') != 'openai/gpt-5.6-terra':
        raise ValueError('Expected the approved GPT-5.6 Terra viewer')
    if manifest['frame_count'] != 500 or [f['sequence_index'] for f in manifest['frames']] != list(range(500)):
        raise ValueError('Only a complete, ordered 500-frame run may replace the primary page')
    if manifest['mask_count'] != sum(len(f['instances']) for f in manifest['frames']):
        raise ValueError('Mask count mismatch')
    validation = json.loads((source/'projection_validation.json').read_text())
    if validation['frame_count'] != 500 or not validation['all_baseline_rgb_exact'] or validation['max_baseline_replay_error_m'] > 1e-5:
        raise ValueError('Projection validation failed')
    published = rebase_manifest(manifest, source, destination)
    old = json.loads((destination/'manifest.json').read_text())
    archive = destination.parent/'yoloe26_gt_raw'
    if old.get('method') != 'gpt':
        if archive.exists():
            raise FileExistsError('Baseline archive already exists; refusing to replace it: '+str(archive))
        archived = rebase_manifest(old, destination, archive)
        archived['method'] = 'yoloe_sam31'
        archive.mkdir()
        for name in ('index.html', 'viewer.js', 'styles.css', 'projection_report.json'):
            shutil.copy2(destination/name, archive/name)
        if (destination/'bridge.js').exists():
            shutil.copy2(destination/'bridge.js', archive/'bridge.js')
        write_manifest(archive, archived)
    if not (archive/'manifest.json').is_file():
        raise FileNotFoundError('Baseline archive is required before replacing the primary page')
    # Reports must keep pointing to the baseline report after the primary report changes.
    report_paths = [source/'projection_report.json', source/'comparison.json']
    comparisons = destination.parents[2]/'comparisons'
    if comparisons.is_dir():
        report_paths.extend(comparisons.glob('*/comparison.json'))
    for path in report_paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if data.get('baseline_report') == str(destination/'projection_report.json'):
            data['baseline_report'] = str(archive/'projection_report.json')
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
            if path.name == 'comparison.json' and path.with_suffix('.js').exists():
                path.with_suffix('.js').write_text('window.METHOD_COMPARISON=' +
                    json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+';\n')
    # Retain destination/assets and destination/frames: the original comparison links to them.
    names = ('viewer.js', 'bridge.js', 'styles.css', 'projection_report.json',
             'projection_validation.json', 'manifest.json', 'manifest.js', 'index.html')
    with tempfile.TemporaryDirectory(prefix='.gpt-publish-', dir=destination.parent) as temp:
        staging = Path(temp)
        for name in names:
            if not name.startswith('manifest.'):
                shutil.copy2(source/name, staging/name)
        write_manifest(staging, published)
        for name in names:
            os.replace(staging/name, destination/name)
    report = {'source': str(source), 'primary_page': str(destination/'index.html'),
              'baseline_archive': str(archive/'index.html'), 'frame_count': 500,
              'mask_count': manifest['mask_count'], 'old_assets_retained': True}
    (destination/'publication.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
