"""Generate one SAM3.1 text-prompt mask in the separate SAM3.1 environment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback
import uuid


def select_candidates(outputs, width, height, threshold):
    """Validate the predictor contract and rank individual, nonempty instances."""
    import numpy as np

    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    scores = np.asarray(outputs["out_probs"], dtype=float).reshape(-1)
    boxes = np.asarray(outputs["out_boxes_xywh"], dtype=float).reshape(-1, 4)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if not len(scores):
        raise ValueError("SAM3.1 found no object; try a different prompt or a lower --sam31-threshold")
    if masks.shape != (len(scores), height, width) or len(boxes) != len(scores):
        raise ValueError("SAM3.1 mask dimensions/cardinality do not match the source image")
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("SAM3.1 returned nonfinite scores or boxes")
    indices = [i for i in range(len(scores)) if masks[i].any() and scores[i] >= threshold]
    indices.sort(key=lambda i: (-scores[i], i))
    if not indices:
        raise ValueError("SAM3.1 returned no nonempty mask above the score threshold")
    candidates = []
    for rank, index in enumerate(indices):
        x, y, w, h = boxes[index]
        candidates.append({
            "instance_index": rank, "model_output_index": index,
            "score": float(scores[index]), "mask_pixels": int(masks[index].sum()),
            "box_xyxy": [float(x * width), float(y * height), float((x + w) * width), float((y + h) * height)],
            "mask_file": f"candidates/{rank:03d}.png",
        })
    return masks, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--instance-index", type=int, default=0)
    args = parser.parse_args()
    if not args.prompt.strip() or not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1 or args.instance_index < 0:
        parser.error("require a nonempty prompt, threshold in [0, 1], and nonnegative instance index")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "model": "SAM3.1 Multiplex", "prompt": args.prompt,
              "image": str(args.image.resolve()), "checkpoint": str(args.checkpoint.resolve()),
              "repo": str(args.repo.resolve()), "threshold": args.threshold,
              "selection_policy": "nonempty masks sorted by descending score; ties by model output order",
              "requested_instance_index": args.instance_index}
    start = time.monotonic()
    try:
        for path in (args.image, args.checkpoint, args.repo / "sam3/model_builder.py"):
            if not path.is_file():
                raise FileNotFoundError(path)
        import numpy as np
        from PIL import Image, ImageDraw
        import torch

        sys.path.insert(0, str(args.repo.resolve()))
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        image = Image.open(args.image).convert("RGB")
        width, height = image.size
        report.update(source_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                      image_size=[width, height], python=sys.version, torch=torch.__version__)
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(args.checkpoint.resolve()), max_num_objects=16, multiplex_count=16,
            use_fa3=False, use_rope_real=False, compile=False, warm_up=False,
            async_loading_frames=False, default_output_prob_thresh=args.threshold,
        )
        report["load_seconds"] = time.monotonic() - start
        with tempfile.TemporaryDirectory(prefix="object_generation_sam31_") as frames:
            image.save(Path(frames) / "000000.jpg", quality=100, subsampling=0)
            # Match InstanceRecognition's adapter: this checkout's public start_session
            # passes offload_state_to_cpu, which the multiplex init_state rejects.
            state = predictor.model.init_state(resource_path=frames, offload_video_to_cpu=False, async_loading_frames=False)
            session = str(uuid.uuid4())
            now = time.time()
            predictor._all_inference_states[session] = {
                "state": state, "session_id": session, "start_time": now, "last_use_time": now,
            }
            try:
                outputs = predictor.handle_request({
                    "type": "add_prompt", "session_id": session, "frame_index": 0,
                    "text": args.prompt, "output_prob_thresh": args.threshold,
                })["outputs"]
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session})
        masks, candidates = select_candidates(outputs, width, height, args.threshold)
        report["candidates"] = candidates
        (args.output / "candidates").mkdir(exist_ok=True)
        for candidate in candidates:
            Image.fromarray(masks[candidate["model_output_index"]].astype(np.uint8) * 255).save(args.output / candidate["mask_file"])
        if args.instance_index >= len(candidates):
            raise ValueError(f"--sam31-instance-index {args.instance_index} is out of range; found {len(candidates)} candidates")
        selected = candidates[args.instance_index]
        mask = Image.fromarray(masks[selected["model_output_index"]].astype(np.uint8) * 255)
        mask_path = args.output / "mask.png"
        mask.save(mask_path)
        overlay = Image.composite(Image.blend(image, Image.new("RGB", image.size, (30, 220, 80)), 0.4), image, mask)
        ImageDraw.Draw(overlay).rectangle(selected["box_xyxy"], outline=(255, 80, 40), width=3)
        overlay.save(args.output / "mask_overlay.png")
        report.update(status="success", candidate_count=len(candidates), selected=selected,
                      mask=str(mask_path), mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                      peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30)
        print(f"Selected instance {args.instance_index} of {len(candidates)} (score={selected['score']:.4f})", flush=True)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        (args.output / "segmentation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
