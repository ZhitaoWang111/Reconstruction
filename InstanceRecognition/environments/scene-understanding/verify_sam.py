"""Verify the installed SAM runtime, optionally running one real JPEG/text inference."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config
from src.inference.io import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='Original JPEG for a SAM-only smoke test')
    parser.add_argument('--prompt', default='office chair')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new verification output directory')
    forbidden = ['ultralytics', 'yolo', 'yolov5', 'yolov8', 'open3d', 'spconv-cu126',
                 'pytorch3d', 'jupyter', 'notebook', 'tensorboard', 'opencv-python', 'scikit-image']
    installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions()}
    unexpected = sorted(set(forbidden) & installed.keys())
    if unexpected:
        raise RuntimeError(f'Unexpected packages: {unexpected}')
    import torch
    import sam3
    from src.inference.sam31_segment import load_predictor, segment
    from src.inference.gpt_sam31 import write_display_artifacts
    assert torch.cuda.is_available(), 'CUDA unavailable'
    assert Path(sam3.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), sam3.__file__
    report = {
        'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'sam3_file': sam3.__file__,
        'cuda_tensor_check': float((torch.ones(4, device='cuda') * 2).sum()),
        'forbidden_packages_present': unexpected,
        'gpt_called': False, 'smoke': None,
    }
    if args.image:
        source = args.image.resolve(strict=True)
        args.output.mkdir(parents=True)
        started = time.monotonic()
        predictor = load_predictor(config.SAM31_CHECKPOINT)
        loaded = time.monotonic()
        try:
            categories = [{'category': args.prompt, 'sam_prompt': args.prompt, 'certainty': 'clear',
                           'visible_scope': 'partial', 'evidence': 'Manual environment verification prompt',
                           'ambiguity': ''}]
            prediction = segment(predictor, source, categories, args.output)
            write_display_artifacts(prediction, args.output)
            write_json(args.output / 'predictions.json', prediction)
            if prediction['instance_count'] < 1:
                raise RuntimeError('No retained masks; inspect this smoke result before calling it verified')
            from PIL import Image
            for item in prediction['instances']:
                with Image.open(args.output / item['mask_file']) as mask:
                    assert mask.getbbox() is not None
            report['smoke'] = {
                'source': str(source), 'source_sha256': sha256(source), 'prompt': args.prompt,
                'checkpoint': str(config.SAM31_CHECKPOINT),
                'model_load_seconds': loaded - started, 'total_seconds': time.monotonic() - started,
                'instance_count': prediction['instance_count'],
                'mask_pixels': [i['mask_pixels'] for i in prediction['instances']],
                'cv2_imported': 'cv2' in sys.modules, 'skimage_imported': 'skimage' in sys.modules,
            }
        finally:
            predictor.shutdown()
    write_json(args.output / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
