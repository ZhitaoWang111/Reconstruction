"""Contracts for the assembler that bridges perception output to a liftable source.

The assembler's whole value is refusal: it is the one place that can tell whether
a given set of masks was actually produced from a given set of labels, because
each mask records the SHA-256 of the label file it came from. Once a lift has run,
a mis-paired source is indistinguishable from a correct one -- the points look
fine and the categories are simply wrong. So most of what is tested here is the
pair of mismatches that must stop the run rather than produce output.

Everything runs against a small fixture, so no GPU, no gateway and no 216 MB of
masks are involved.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.geometry import assemble_perception_source as assembler
from src.geometry.backproject_masks import write_ply


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_backdrop(path, count=3):
    points = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    write_ply(path, points, np.full((count, 3), 200, np.uint8),
              np.zeros(count, np.uint32), np.zeros(count, np.uint32),
              np.arange(count, dtype=np.uint32))
    return path


def instance(instance_id=1, class_name="office chair", index=0):
    return {
        "raw_candidate_id": instance_id, "instance_id": instance_id,
        "class_id": index, "prompt_index": index, "class_name": class_name,
        "sam_prompt": class_name, "sam_prompt_type": "text", "gpt_certainty": "clear",
        "visible_scope": "whole", "evidence": "visible in the frame", "ambiguity": "",
        "sam_score": 0.9, "sam_box_xyxy": [1.0, 2.0, 3.0, 4.0], "mask_pixels": 4,
        "mask_file": f"raw_masks/{instance_id:03d}_{class_name.replace(' ', '_')}.png",
    }


class Perception:
    """A perception tree whose labels and masks agree unless asked otherwise."""

    def __init__(self, root, frames=2, labels=("table", "office chair"), mismatch=None):
        self.frames = frames
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        (self.frames_dir / "scene0000_00").mkdir(parents=True)
        self.perception = self.root / "artifacts/experiments/test-run/perception"
        self.labels_run = self.perception / "labels"
        self.masks_run = self.perception / "masks"
        for folder in (self.labels_run / "labels", self.masks_run):
            folder.mkdir(parents=True)

        records = []
        for seq in range(frames):
            fid = seq * 11
            rgb = self.frames_dir / "scene0000_00" / f"frame_{fid:06d}.jpg"
            rgb.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6), "white").save(rgb)
            digest = sha(rgb)
            # The recorded source is the pre-refactor location, which no longer
            # exists -- exactly the condition that makes relocation necessary.
            recorded_source = str(self.root / f"runtime/frames/scene0000_00/frame_{fid:06d}.jpg")
            records.append({"scene_id": "scene0000_00", "sequence_index": seq,
                            "source_frame_id": fid, "source": recorded_source,
                            "source_sha256": digest})

            categories = [{"category": name, "sam_prompt": name, "certainty": "clear",
                           "visible_scope": "whole", "evidence": "visible", "ambiguity": ""}
                          for name in labels]
            (self.labels_run / f"labels/{seq:04d}.json").write_text(json.dumps({
                "scene_id": "scene0000_00", "sequence_index": seq, "source_frame_id": fid,
                "source": recorded_source, "source_sha256": digest,
                "requested_model": "openai/gpt-5.6-terra", "elapsed_seconds": 1.0,
                "usage": {"total_tokens": 10}, "attempts": 1, "prior_errors": [],
                "labels": {"categories": categories, "unresolved_regions": []},
            }))

            instances = [instance(i + 1, name, i) for i, name in enumerate(labels)]
            claimed_count = len(instances)
            if mismatch == "count":
                # The masks report more instances than they carry: a summary that
                # disagrees with its own list, which a lift would index past.
                instances.append(instance(len(labels) + 1, "door", len(labels)))
                claimed_count = len(instances) + 1
            if mismatch == "labels_hash":
                labels_sha = "0" * 64
            else:
                labels_sha = sha(self.labels_run / f"labels/{seq:04d}.json")
            if mismatch == "rgb_hash":
                digest = "1" * 64
            sam31 = self.masks_run / f"frame_{fid:06d}/sam31"
            (sam31 / "raw_masks").mkdir(parents=True)
            for item in instances:
                Image.new("L", (8, 6), 255).save(sam31 / item["mask_file"])
            Image.new("RGB", (8, 6), "white").save(sam31 / "annotated.jpg")
            (sam31 / "predictions.json").write_text(json.dumps({
                "sequence_index": seq, "source_frame_id": fid, "source": recorded_source,
                "instance_count": claimed_count, "instances": instances,
                "gpt_model": "openai/gpt-5.6-terra",
                "provenance": {"labels_sha256": labels_sha, "source_sha256": digest},
            }))
        (self.labels_run / "config.json").write_text(json.dumps({
            "scene_id": "scene0000_00", "model": "openai/gpt-5.6-terra", "frames": records}))

    def assemble(self, output, **kwargs):
        return assembler.build("test-run", output, frames_dir=self.frames_dir,
                               backdrop=write_backdrop(self.root / "backdrop.ply"), **kwargs)


@pytest.fixture(autouse=True)
def artifacts_root(monkeypatch, tmp_path):
    """perception_dir() resolves under the artifacts root, so point it at the fixture."""
    monkeypatch.setattr(assembler.config, "ARTIFACTS_ROOT", tmp_path / "artifacts")


def test_assemble_names_every_frame_and_mask(tmp_path):
    perception = Perception(tmp_path)
    output = perception.assemble(tmp_path / "source")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["frame_count"] == 2 and manifest["mask_count"] == 4
    assert [f["source_frame_id"] for f in manifest["frames"]] == [0, 11]
    assert manifest["model"] == "openai/gpt-5.6-terra"


def test_every_path_resolves_from_the_output(tmp_path):
    """The lift resolves these itself, so a wrong relative base is a broken source."""
    output = Perception(tmp_path).assemble(tmp_path / "source")
    manifest = json.loads((output / "manifest.json").read_text())
    assert (output / manifest["background_asset"]).is_file()
    for frame in manifest["frames"]:
        for key in ("original_rgb_image", "annotated_image", "predictions_json"):
            assert (output / frame[key]).is_file(), key
        for item in frame["instances"]:
            assert (output / item["mask_file"]).is_file(), item["mask_file"]


def test_the_rgb_is_relocated_not_taken_from_the_labels(tmp_path):
    """The recorded `source` path is the pre-refactor one and no longer exists."""
    perception = Perception(tmp_path)
    recorded = json.loads((perception.labels_run / "labels/0000.json").read_text())["source"]
    assert not Path(recorded).is_file()
    output = perception.assemble(tmp_path / "source")
    manifest = json.loads((output / "manifest.json").read_text())
    assert Path(manifest["frames"][0]["original_rgb_image"]).name == "frame_000000.jpg"


def test_labels_from_a_different_run_are_refused(tmp_path):
    """The failure this module exists to catch: masks that pair with other labels."""
    perception = Perception(tmp_path, mismatch="labels_hash")
    with pytest.raises(ValueError, match="do not belong together"):
        perception.assemble(tmp_path / "source")


def test_an_rgb_that_belongs_to_neither_half_is_refused(tmp_path):
    perception = Perception(tmp_path, mismatch="rgb_hash")
    with pytest.raises(ValueError, match="describe a different RGB"):
        perception.assemble(tmp_path / "source")


def test_instance_count_that_disagrees_with_the_list_is_refused(tmp_path):
    perception = Perception(tmp_path, mismatch="count")
    with pytest.raises(ValueError, match="instance_count disagrees"):
        perception.assemble(tmp_path / "source")


def test_a_missing_rgb_stops_the_run(tmp_path):
    perception = Perception(tmp_path)
    (perception.frames_dir / "scene0000_00/frame_000011.jpg").unlink()
    with pytest.raises(SystemExit, match="Missing RGB"):
        perception.assemble(tmp_path / "source")


def test_the_output_directory_must_be_new(tmp_path):
    perception = Perception(tmp_path)
    perception.assemble(tmp_path / "source")
    with pytest.raises(ValueError, match="new empty output directory"):
        perception.assemble(tmp_path / "source")


def test_a_gap_in_the_sequence_run_is_refused(tmp_path):
    """observations.py requires contiguous sequences starting at zero."""
    perception = Perception(tmp_path, frames=2)
    with pytest.raises(ValueError, match="contiguous run"):
        perception.assemble(tmp_path / "source", first=1)


def test_more_masks_than_the_bit_capacity_is_refused(tmp_path):
    perception = Perception(tmp_path, labels=tuple(f"class {i}" for i in range(33)))
    with pytest.raises(ValueError, match="mask_bits capacity"):
        perception.assemble(tmp_path / "source")


def test_no_lifting_policy_is_claimed(tmp_path):
    """Until a lift has run there is no geometry, so the Harness must refuse it."""
    output = Perception(tmp_path).assemble(tmp_path / "source")
    assert "lifting_policy" not in json.loads((output / "manifest.json").read_text())


def test_the_backdrop_is_encoded_for_the_viewer(tmp_path):
    """The field is a script the browser loads, not a PLY it could read."""
    output = Perception(tmp_path).assemble(tmp_path / "source")
    asset = (output / "assets/background.js").read_text()
    assert asset.startswith('window.registerMaskCloud("background",')
    assert json.loads((output / "manifest.json").read_text())["background_asset"] == \
        "assets/background.js"


def test_bounds_cover_the_backdrop_the_viewer_is_framed_on(tmp_path):
    output = Perception(tmp_path).assemble(tmp_path / "source")
    manifest = json.loads((output / "manifest.json").read_text())
    points, _ = assembler.read_point_ply(tmp_path / "backdrop.ply")
    assert np.allclose(manifest["bounds"]["minimum"], points.min(axis=0))
    assert np.allclose(manifest["bounds"]["maximum"], points.max(axis=0))


def test_manifest_js_carries_the_same_manifest(tmp_path):
    """viewer.js reads window.MASK_PROJECTION_MANIFEST, not the .json."""
    output = Perception(tmp_path).assemble(tmp_path / "source")
    text = (output / "manifest.js").read_text()
    assert text.startswith("window.MASK_PROJECTION_MANIFEST=")
    assert json.loads(text.split("=", 1)[1].rstrip(";\n")) == \
        json.loads((output / "manifest.json").read_text())


def test_the_report_records_what_was_verified(tmp_path):
    output = Perception(tmp_path).assemble(tmp_path / "source")
    report = json.loads((output / "assembly_report.json").read_text())
    assert report["policy"] == assembler.POLICY
    assert report["mask_count"] == 4 and report["frame_count"] == 2
    assert report["backdrop_points"] == 3
    assert len(report["backdrop_sha256"]) == 64


def test_every_instance_carries_a_display_colour(tmp_path):
    """Viewers key a mask by its colour; a missing one makes every mask white.

    Perception records no colour, so the assembler has to mint one. Without it the
    page falls back to white for both the sidebar swatch and the projected points,
    which is indistinguishable from "all masks are the same object".
    """
    output = Perception(tmp_path).assemble(tmp_path / "source")
    manifest = json.loads((output / "manifest.json").read_text())
    for frame in manifest["frames"]:
        for item in frame["instances"]:
            assert len(item["color"]) == 3
            assert all(0 <= channel <= 255 for channel in item["color"])
            assert item["color"] != [255, 255, 255], (
                "a white fallback is exactly what the page already does")


def test_colours_are_stable_across_rebuilds(tmp_path):
    """The hue is derived, not sampled, so a rebuild must not reshuffle masks."""
    perception = Perception(tmp_path)
    first = perception.assemble(tmp_path / "source-a")
    second = perception.assemble(tmp_path / "source-b")
    colours = lambda out: [i["color"] for f in json.loads(
        (out / "manifest.json").read_text())["frames"] for i in f["instances"]]
    assert colours(first) == colours(second)


def test_labels_and_masks_are_referenced_in_place(tmp_path):
    """The masks are 216 MB; a copy per experiment is not acceptable."""
    output = Perception(tmp_path).assemble(tmp_path / "source")
    manifest = json.loads((output / "manifest.json").read_text())
    resolved = (output / manifest["frames"][0]["instances"][0]["mask_file"]).resolve()
    assert resolved.is_relative_to(tmp_path / "artifacts")
    assert not (output / "frames").exists()
