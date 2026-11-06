"""Contracts for the SAM3.1 stage's orchestration.

`GptSam31Segmenter` is replaced by a stub so these run without torch: what is
being tested is the bookkeeping around the model -- which frames are processed,
whether a cached frame is still valid, and the shape of what lands on disk.
The last is the point. `predictions.json` is read field-by-field by the page
builder, the harness observations and the geometry modules, so a rename breaks
those quietly; the schema test here is the tripwire.
"""
import hashlib
import json
from pathlib import Path

import pytest

from src.inference import gpt_sam31

# Every key a downstream consumer reads, with the consumer that needs it.
CONSUMED_FIELDS = {
    "mask_file",        # geometry: amb3r_mask_source / observations
    "class_name",       # observations -> category_label; page legend
    "sam_prompt",       # page: prompt shown for the mask
    "sam_prompt_type",  # page: "text" vs the box-prompt fallbacks
    "sam_score",        # page: ranking
    "gpt_certainty",    # page: how sure GPT was
    "evidence",         # page: why GPT proposed it
    "visible_scope",
    "ambiguity",
    "mask_pixels",
    "instance_id",      # page + per-frame npz address masks by this
    "raw_candidate_id",  # audit: pairs a prompt with the candidates it produced
}


def labels_run(tmp_path, frames=1):
    """A labeling run directory with the config.json and labels/ the CLI reads."""
    run = tmp_path / "labels"
    (run / "labels").mkdir(parents=True)
    records = []
    for sequence in range(frames):
        source = tmp_path / f"frames/frame_{sequence:06d}.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"jpeg-{sequence}".encode())
        records.append({"scene_id": "scene0000_00", "sequence_index": sequence,
                        "source_frame_id": sequence, "source": str(source),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        (run / f"labels/{sequence:04d}.json").write_text(json.dumps({
            "source": str(source), "source_frame_id": sequence,
            "source_sha256": records[-1]["source_sha256"],
            "requested_model": "openai/gpt-5.6-terra", "elapsed_seconds": 1.0,
            "usage": {"total_tokens": 10},
            "labels": {"categories": [
                {"category": "table", "sam_prompt": "table", "certainty": "clear",
                 "visible_scope": "whole", "evidence": "flat top", "ambiguity": ""}],
                "unresolved_regions": []},
        }))
    (run / "config.json").write_text(json.dumps({"frames": records}))
    return run


class StubSegmenter:
    """Stands in for the model: records construction so reuse is observable."""

    built = 0

    def __init__(self, checkpoint, threshold=gpt_sam31.DEFAULT_THRESHOLD):
        StubSegmenter.built += 1
        self.checkpoint, self.threshold = checkpoint, threshold
        self.load_seconds = 0.5

    def infer(self, source, labels_path, output, max_masks=gpt_sam31.DEFAULT_MAX_MASKS,
              min_area_fraction=gpt_sam31.DEFAULT_MIN_MASK_AREA_FRACTION):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "raw_masks").mkdir(exist_ok=True)
        mask = Path(output) / "raw_masks/001_table.png"
        mask.write_bytes(b"png")
        categories = json.loads(Path(labels_path).read_text())["labels"]["categories"]
        instance = {
            "raw_candidate_id": 1, "instance_id": 1, "class_name": categories[0]["category"],
            "class_id": 0, "prompt_index": 0, "sam_prompt": categories[0]["sam_prompt"],
            "gpt_certainty": "clear", "visible_scope": "whole", "evidence": "flat top",
            "ambiguity": "", "sam_score": 0.9, "sam_box_xyxy": [1.0, 2.0, 3.0, 4.0],
            "sam_prompt_type": "text", "mask_file": "raw_masks/001_table.png",
            "mask_pixels": 5,
        }
        return {"stage": "stub", "source": str(source), "image_shape_hwc": [968, 1296, 3],
                "instance_count": 1, "instances": [instance], "raw_candidates": [instance],
                "raw_candidate_count": 1, "excluded_by_budget": [],
                "prompts": [{"prompt_index": 0, "raw_candidate_ids": [1]}],
                "unique_classes": [instance["class_name"]],
                "policy": {"temporal_propagation": False}, "elapsed_seconds": 0.1}

    def close(self):
        pass


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "sam3.1_multiplex.pt"
    path.write_bytes(b"weights")
    return path


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    StubSegmenter.built = 0
    monkeypatch.setattr(gpt_sam31, "GptSam31Segmenter", StubSegmenter)


def test_every_consumed_field_is_written(tmp_path, checkpoint):
    """The tripwire: a field rename here silently breaks the page and the harness."""
    run = labels_run(tmp_path)
    gpt_sam31.run(run, tmp_path / "out", checkpoint)
    prediction = json.loads((tmp_path / "out/frame_000000/sam31/predictions.json").read_text())
    assert CONSUMED_FIELDS <= set(prediction["instances"][0])
    assert prediction["instance_count"] == len(prediction["instances"])
    assert prediction["raw_candidate_count"] == len(prediction["raw_candidates"])
    assert prediction["prompts"][0]["raw_candidate_ids"] == [1]


def test_output_keeps_the_frame_directory_shape(tmp_path, checkpoint):
    """The page and the mask sources resolve mask_file relative to the run root."""
    run = labels_run(tmp_path)
    gpt_sam31.run(run, tmp_path / "out", checkpoint)
    out = tmp_path / "out/frame_000000/sam31"
    prediction = json.loads((out / "predictions.json").read_text())
    for item in prediction["raw_candidates"]:
        assert (out / item["mask_file"]).is_file()
    assert (tmp_path / "out/summary.json").is_file()


def test_the_model_is_built_once_for_many_frames(tmp_path, checkpoint):
    """Loading 3.3 GiB per frame would dominate the run, so this is a contract."""
    run = labels_run(tmp_path, frames=3)
    gpt_sam31.run(run, tmp_path / "out", checkpoint)
    assert StubSegmenter.built == 1
    assert len(json.loads((tmp_path / "out/summary.json").read_text())["frames"]) == 3


def test_cached_frame_is_reused_without_rebuilding(tmp_path, checkpoint):
    run = labels_run(tmp_path)
    gpt_sam31.run(run, tmp_path / "out", checkpoint)
    first = json.loads((tmp_path / "out/frame_000000/sam31/predictions.json").read_text())
    first["sentinel"] = "kept"
    (tmp_path / "out/frame_000000/sam31/predictions.json").write_text(json.dumps(first))

    StubSegmenter.built = 0
    gpt_sam31.run(run, tmp_path / "out", checkpoint)
    assert StubSegmenter.built == 0
    assert json.loads((tmp_path / "out/frame_000000/sam31/predictions.json").read_text())["sentinel"] == "kept"


def test_cache_is_refused_when_the_frame_image_changes(tmp_path, checkpoint):
    """A replaced frame must not keep the previous answer's masks."""
    run = labels_run(tmp_path)
    gpt_sam31.run(run, tmp_path / "out", checkpoint)
    (tmp_path / "frames/frame_000000.jpg").write_bytes(b"different-bytes")
    with pytest.raises(ValueError, match="different inputs"):
        gpt_sam31.run(run, tmp_path / "out", checkpoint)


def test_cache_is_refused_when_the_threshold_changes(tmp_path, checkpoint):
    run = labels_run(tmp_path)
    gpt_sam31.run(run, tmp_path / "out", checkpoint)
    with pytest.raises(ValueError, match="different inputs"):
        gpt_sam31.run(run, tmp_path / "out", checkpoint, threshold=0.9)


def test_missing_labels_stop_the_run(tmp_path, checkpoint):
    """A frame without labels is a missing input, not a frame to skip."""
    run = labels_run(tmp_path, frames=2)
    (run / "labels/0001.json").unlink()
    with pytest.raises(SystemExit, match="Missing GPT labels"):
        gpt_sam31.run(run, tmp_path / "out", checkpoint)


def test_a_source_frame_mismatch_is_refused(tmp_path, checkpoint):
    run = labels_run(tmp_path)
    labels = json.loads((run / "labels/0000.json").read_text())
    labels["source_frame_id"] = 99
    (run / "labels/0000.json").write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="disagree on frame"):
        gpt_sam31.run(run, tmp_path / "out", checkpoint)


def test_provenance_pins_the_frame_labels_and_checkpoint(tmp_path, checkpoint):
    run = labels_run(tmp_path)
    gpt_sam31.run(run, tmp_path / "out", checkpoint)
    provenance = json.loads(
        (tmp_path / "out/frame_000000/sam31/predictions.json").read_text())["provenance"]
    assert set(provenance) >= {"source_sha256", "labels_sha256", "checkpoint",
                              "checkpoint_size", "checkpoint_mtime_ns"}
    assert provenance["source_sha256"] == hashlib.sha256((tmp_path / "frames/frame_000000.jpg").read_bytes()).hexdigest()
    assert provenance["labels_sha256"] == hashlib.sha256((run / "labels/0000.json").read_bytes()).hexdigest()


def test_selection_and_labels_survive_the_gateway_metadata(tmp_path, checkpoint):
    """GPT's own report is carried through, not recomputed."""
    run = labels_run(tmp_path)
    gpt_sam31.run(run, tmp_path / "out", checkpoint)
    prediction = json.loads((tmp_path / "out/frame_000000/sam31/predictions.json").read_text())
    assert prediction["gpt_model"] == "openai/gpt-5.6-terra"
    assert prediction["unresolved_regions"] == []
    assert prediction["sam_model_load_seconds"] == 0.5


def test_cli_rejects_invalid_mask_settings(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["gpt_sam31", "--run", "test", "--max-masks", "99"])
    with pytest.raises(SystemExit):
        gpt_sam31.parse_args()
