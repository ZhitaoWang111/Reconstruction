"""Batch GPT labels -> SAM3.1 masks. Finish labeling before running this stage."""
import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src import config
from .io import perception_dir, read_json, sha256, write_json
from .label_frames import validate
from .sam31_segment import (DEFAULT_MAX_MASKS, DEFAULT_MIN_MASK_AREA_FRACTION, DEFAULT_THRESHOLD,
                            PALETTE, load_predictor, segment)


def write_display_artifacts(prediction, output):
    output = Path(output)
    with Image.open(prediction['source']) as image:
        rgb = np.array(image.convert('RGB'))
    preview = rgb.astype(np.float32)
    ids = np.zeros(rgb.shape[:2], dtype=np.uint16)
    for index, item in enumerate(prediction['instances']):
        with Image.open(output / item['mask_file']) as image:
            mask = np.asarray(image) > 0
        if mask.shape != ids.shape:
            raise ValueError('Mask dimensions differ from RGB')
        preview[mask] = 0.46 * preview[mask] + 0.54 * np.asarray(PALETTE[index % len(PALETTE)])
        ids[mask] = item['instance_id']
    overlay = Image.fromarray(preview.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for index, item in enumerate(prediction['instances']):
        box = item['sam_box_xyxy']
        draw.rectangle(box, outline=PALETTE[index % len(PALETTE)], width=2)
        draw.text((max(0, box[0]), max(0, box[1] - 12)),
                  f'{item["instance_id"]} {item["class_name"]}', fill='white', stroke_width=1, stroke_fill='black')
    overlay.save(output / 'annotated.jpg')
    Image.fromarray(ids).save(output / 'instance_id_map.png')
    prediction['artifacts'] = {'annotated_image': 'annotated.jpg', 'instance_id_map': 'instance_id_map.png'}
    prediction['instance_id_map_policy'] = 'display-only; later instances overwrite overlaps; binary masks are authoritative'
    prediction['timing'] = {'segment_seconds': prediction['elapsed_seconds']}
    return prediction


class GptSam31Segmenter:
    def __init__(self, checkpoint, threshold=DEFAULT_THRESHOLD):
        started = time.monotonic()
        self.predictor = load_predictor(checkpoint, threshold)
        self.threshold = threshold
        self.load_seconds = time.monotonic() - started

    def infer(self, source, labels_path, output, max_masks=DEFAULT_MAX_MASKS,
              min_area_fraction=DEFAULT_MIN_MASK_AREA_FRACTION):
        labels = read_json(labels_path)
        result = segment(self.predictor, source, labels['labels']['categories'], output,
                         self.threshold, max_masks, min_area_fraction)
        return write_display_artifacts(result, output)

    def close(self):
        self.predictor.shutdown()


def frame_provenance(record, labels_path, checkpoint, threshold, max_masks,
                     min_area_fraction=DEFAULT_MIN_MASK_AREA_FRACTION):
    checkpoint = Path(checkpoint).resolve(strict=True)
    stat = checkpoint.stat()
    return {'format': 'gpt_sam31_v2', 'source_sha256': sha256(record['source']),
            'labels_sha256': sha256(labels_path), 'threshold': threshold, 'max_masks': max_masks,
            'min_mask_area_fraction': min_area_fraction,
            'checkpoint': str(checkpoint), 'checkpoint_size': stat.st_size,
            'checkpoint_mtime_ns': stat.st_mtime_ns}


def run(labels_run, output, checkpoint, threshold=DEFAULT_THRESHOLD, max_masks=DEFAULT_MAX_MASKS,
        min_area_fraction=DEFAULT_MIN_MASK_AREA_FRACTION):
    if not 0 <= threshold <= 1 or not 1 <= max_masks <= 32:
        raise ValueError('Invalid mask settings')
    if not 0 <= min_area_fraction < 1:
        raise ValueError('Invalid minimum mask area fraction')
    labels_run, output = Path(labels_run).resolve(), Path(output).resolve()
    records = read_json(labels_run / 'config.json')['frames']
    if not records:
        raise ValueError('No frames selected')
    if len({r['source_frame_id'] for r in records}) != len(records):
        raise ValueError('Duplicate source frame IDs')
    segmenter, results = None, []
    try:
        for record in records:
            seq, fid = record['sequence_index'], record['source_frame_id']
            label_path = labels_run / f'labels/{seq:04d}.json'
            if not label_path.is_file():
                raise SystemExit(f'Missing GPT labels: {label_path}')
            labels = read_json(label_path)
            if labels['source_frame_id'] != fid or Path(labels['source']).resolve() != Path(record['source']).resolve():
                raise ValueError(f'GPT labels and batch disagree on frame {seq}')
            validate(labels['labels'])
            provenance = frame_provenance(record, label_path, checkpoint, threshold, max_masks,
                                          min_area_fraction)
            folder = output / f'frame_{fid:06d}' / 'sam31'
            prediction_path = folder / 'predictions.json'
            if folder.exists():
                prediction = read_json(prediction_path)
                if prediction.get('provenance') != provenance:
                    raise ValueError(f'Cached frame {fid} has different inputs; choose a new --run')
                for name, digest in prediction['artifact_sha256'].items():
                    if not (folder / name).is_file() or sha256(folder / name) != digest:
                        raise ValueError(f'Cached mask artifact is missing or changed: {folder / name}')
            else:
                if labels['source_sha256'] != provenance['source_sha256'] or record['source_sha256'] != provenance['source_sha256']:
                    raise ValueError(f'Frame {seq} has different inputs from its GPT labels')
                if segmenter is None:
                    segmenter = GptSam31Segmenter(checkpoint, threshold)
                folder.parent.mkdir(parents=True, exist_ok=True)
                # Publish the whole frame only after inference and all artifact writes succeed.
                with tempfile.TemporaryDirectory(prefix='.sam31-', dir=folder.parent) as temporary:
                    staging = Path(temporary) / 'sam31'
                    staging.mkdir()
                    prediction = segmenter.infer(record['source'], label_path, staging, max_masks,
                                                 min_area_fraction)
                    prediction.update({
                        'sequence_index': seq, 'source_frame_id': fid, 'provenance': provenance,
                        'gpt_model': labels['requested_model'], 'gpt_elapsed_seconds': labels['elapsed_seconds'],
                        'gpt_usage': labels['usage'], 'unresolved_regions': labels['labels']['unresolved_regions'],
                        'sam_model_load_seconds': segmenter.load_seconds,
                    })
                    files = {item['mask_file'] for item in prediction['raw_candidates']}
                    files.update(prediction.get('artifacts', {}).values())
                    prediction['artifact_sha256'] = {name: sha256(staging / name) for name in sorted(files)}
                    write_json(staging / 'predictions.json', prediction)
                    staging.rename(folder)
            results.append({'sequence_index': seq, 'source_frame_id': fid,
                            'prediction': str(prediction_path), 'instance_count': prediction['instance_count'],
                            'raw_candidate_count': prediction['raw_candidate_count']})
            print(f'MASKED {seq}: {prediction["instance_count"]} instances', flush=True)
    finally:
        if segmenter is not None:
            segmenter.close()
    write_json(output / 'summary.json', {'frame_count': len(results), 'frames': results,
               'labels_config': str(labels_run / 'config.json'), 'checkpoint': str(Path(checkpoint).resolve())})
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Experiment name used by label_frames')
    parser.add_argument('--checkpoint', type=Path, default=config.SAM31_CHECKPOINT)
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--max-masks', type=int, default=DEFAULT_MAX_MASKS)
    parser.add_argument('--min-mask-area', type=float, default=DEFAULT_MIN_MASK_AREA_FRACTION,
                        help='Drop masks covering less than this fraction of the frame (0 disables)')
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1 or not 1 <= args.max_masks <= 32:
        parser.error('Require threshold in 0..1 and max-masks in 1..32')
    if not 0 <= args.min_mask_area < 1:
        parser.error('Require min-mask-area in 0..1')
    return args


def main(argv=None):
    args = parse_args(argv)
    folder = perception_dir(args.run)
    return run(folder / 'labels', folder / 'masks', args.checkpoint, args.threshold, args.max_masks,
               args.min_mask_area)


if __name__ == '__main__':
    main()
