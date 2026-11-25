"""M9 evaluation: how well the assembled scene matches the scan and the photos, and what the run cost.

    from eval.run import evaluate
    report = evaluate(run_dir)          # writes eval.json and report.html into run_dir/eval

Read-only: it never changes a scene. Four parts:
  * holdout-frame silhouette IoU (the pilot's metric, copied from src/legacy/evaluate.py),
    on frames the agent could not have seen: every sampled frame where ScanNet's 2D instance
    annotation covers the object, minus every object's first key frame (the frames any batch
    can share). Frames are picked per object with a seed from its id, so the set no longer
    depends on the batching;
  * baselines scored on the same frames with the same renderer: B0 (scan box, yaw 0),
    B1 (raw SAM3D orientation), M3 (the agent's starting pose);
  * geometry and physics of the final state (M4's measurements, the final gravity run);
  * process numbers per batch (turns, tokens, cost, undos, reopens).
"""
from .run import evaluate

__all__ = ['evaluate']
