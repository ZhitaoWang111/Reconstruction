"""Contracts for the GPT labeling stage.

Every test here replaces `urllib.request.urlopen`, so the suite never touches the
gateway and needs no key. What is worth protecting is the pair of decisions that
make a batch resumable: a reply must satisfy the schema before anything stores
it, and a cached reply must be reused only when the frame and model it was
produced from are still the same.
"""
import io
import hashlib
import json
import urllib.error
from pathlib import Path

import pytest

from src.inference import label_frames

LABELS = {"categories": [
    {"category": "table", "sam_prompt": "table", "certainty": "clear",
     "visible_scope": "whole", "evidence": "flat top on legs", "ambiguity": ""},
], "unresolved_regions": []}


def record(sequence=0, digest="hash-a", source="/tmp/frame.jpg"):
    return {"scene_id": "scene0000_00", "sequence_index": sequence, "source_frame_id": sequence,
            "source": source, "source_sha256": digest}


def reply(content, finish_reason="stop"):
    return {"id": "resp-1", "model": "openai/gpt-5.6-terra",
            "choices": [{"finish_reason": finish_reason,
                         "message": {"content": json.dumps(content)}}],
            "usage": {"total_tokens": 10}}


def fake_gateway(monkeypatch, responses):
    """Serve `responses` in order; an entry that is an exception is raised."""
    seen = []
    monkeypatch.setattr(label_frames.time, "sleep", lambda _: None)

    def urlopen(request, timeout):
        seen.append(json.loads(request.data))
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        handle = io.BytesIO(json.dumps(item).encode())
        handle.headers = {"Content-Type": "application/json"}
        return handle

    monkeypatch.setattr(label_frames.urllib.request, "urlopen", urlopen)
    return seen


def call(record_item, output, monkeypatch, responses, source_bytes=b"jpeg-bytes", force=False):
    """Label one frame against a fake gateway. The source image must really exist,
    because the request body carries its bytes."""
    record_item = dict(record_item)
    if record_item['source'] == '/tmp/frame.jpg':
        record_item['source'] = str(output / 'input/frame.jpg')
    source = Path(record_item["source"])
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(source_bytes)
    record_item['source_sha256'] = hashlib.sha256(source_bytes).hexdigest()
    seen = fake_gateway(monkeypatch, responses)
    result = label_frames.label_one(record_item, output, "openai/gpt-5.6-terra",
                                    "https://gateway.invalid/v1", "secret-key", "PROMPT",
                                    force=force)
    return result, seen


def test_successful_label_is_stored_with_provenance(tmp_path, monkeypatch):
    result, _ = call(record(), tmp_path, monkeypatch, [reply(LABELS)])
    assert result["labels"]["categories"][0]["category"] == "table"
    stored = json.loads((tmp_path / "labels/0000.json").read_text())
    assert stored["source_sha256"] == hashlib.sha256(b'jpeg-bytes').hexdigest()
    assert stored["requested_model"] == "openai/gpt-5.6-terra"


def test_api_key_never_reaches_disk(tmp_path, monkeypatch):
    """A gateway error body can echo the request; the log must not keep the key."""
    failure = urllib.error.HTTPError("https://gateway.invalid/v1", 500, "boom", {},
                                     io.BytesIO(b"rejected secret-key"))
    call(record(), tmp_path, monkeypatch, [failure, failure, failure])
    errors = json.loads((tmp_path / "errors/0000.json").read_text())
    assert not any("secret-key" in message for message in errors["error"])
    assert any("[REDACTED]" in message for message in errors["error"])


def test_cached_labels_are_reused_without_a_request(tmp_path, monkeypatch):
    first, _ = call(record(), tmp_path, monkeypatch, [reply(LABELS)])
    second, seen = call(record(), tmp_path, monkeypatch, [])
    assert seen == [] and second == first


@pytest.mark.parametrize("changed", [
    {"source_sha256": "hash-b"},              # the frame image was replaced
    {"requested_model": "openai/other"},      # a different model produced it
    {"prompt_sha256": "a-different-prompt"},  # the question asked of it changed
])
def test_cache_is_refused_when_provenance_moved(tmp_path, monkeypatch, changed):
    call(record(), tmp_path, monkeypatch, [reply(LABELS)])
    path = tmp_path / "labels/0000.json"
    stored = json.loads(path.read_text())
    stored.update(changed)
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="different image, model or prompt"):
        call(record(), tmp_path, monkeypatch, [])


def test_cached_result_records_the_prompt_it_answered(tmp_path, monkeypatch):
    """Without this, a prompt change cannot be detected at the per-frame level."""
    result, _ = call(record(), tmp_path, monkeypatch, [reply(LABELS)])
    import hashlib

    assert result["prompt_sha256"] == hashlib.sha256(b"PROMPT").hexdigest()


def test_force_ignores_the_frame_cache(tmp_path, monkeypatch):
    """An explicit forced call replaces the cached frame."""
    call(record(), tmp_path, monkeypatch, [reply(LABELS)])
    result, seen = call(record(), tmp_path, monkeypatch, [reply(LABELS)], force=True)
    assert len(seen) == 1 and result["labels"] == LABELS


def test_a_damaged_cache_entry_is_a_miss_not_a_failure(tmp_path, monkeypatch):
    """An interrupted write must not make the run unrecoverable."""
    call(record(), tmp_path, monkeypatch, [reply(LABELS)])
    (tmp_path / "labels/0000.json").write_text('{"labels": {"categ')  # truncated
    result, seen = call(record(), tmp_path, monkeypatch, [reply(LABELS)])
    assert len(seen) == 1 and result["labels"] == LABELS


def test_an_older_format_cache_entry_is_a_miss(tmp_path, monkeypatch):
    """A cache from a format this version cannot read is regenerated, not fatal."""
    (tmp_path / "labels").mkdir(parents=True)
    (tmp_path / "labels/0000.json").write_text(json.dumps({"unrelated": True}))
    result, seen = call(record(), tmp_path, monkeypatch, [reply(LABELS)])
    assert len(seen) == 1 and "labels" in result


def test_retryable_failure_is_retried_then_succeeds(tmp_path, monkeypatch):
    failure = urllib.error.HTTPError("https://gateway.invalid/v1", 503, "busy", {}, io.BytesIO(b""))
    result, seen = call(record(), tmp_path, monkeypatch, [failure, reply(LABELS)])
    assert len(seen) == 2 and result["attempts"] == 2
    assert result["prior_errors"] and not (tmp_path / "errors/0000.json").exists()


def test_auth_failure_is_not_retried(tmp_path, monkeypatch):
    """A rejected key will be rejected again; retrying only spends it three times."""
    failure = urllib.error.HTTPError("https://gateway.invalid/v1", 401, "denied", {}, io.BytesIO(b""))
    result, seen = call(record(), tmp_path, monkeypatch, [failure])
    assert len(seen) == 1 and "error" in result
    assert (tmp_path / "errors/0000.json").is_file()


def test_incomplete_response_is_not_accepted(tmp_path, monkeypatch):
    """A truncated body parses as JSON but is not a finished answer."""
    result, _ = call(record(), tmp_path, monkeypatch,
                     [reply(LABELS, finish_reason="length")] * 3)
    assert "error" in result and not (tmp_path / "labels/0000.json").exists()


def test_failed_frame_leaves_no_label_file(tmp_path, monkeypatch):
    call(record(), tmp_path, monkeypatch, [reply({"categories": "not a list"})] * 3)
    assert not (tmp_path / "labels/0000.json").exists()


def test_a_failed_forced_rerun_supersedes_the_old_labels(tmp_path, monkeypatch):
    """Otherwise the stale labels still satisfy provenance and pass as this run's."""
    call(record(), tmp_path, monkeypatch, [reply(LABELS)])
    failure = urllib.error.HTTPError("https://gateway.invalid/v1", 500, "boom", {}, io.BytesIO(b""))
    call(record(), tmp_path, monkeypatch, [failure] * 3, force=True)
    assert not (tmp_path / "labels/0000.json").exists()
    superseded = json.loads((tmp_path / "labels/0000.superseded.json").read_text())
    assert superseded["labels"] == LABELS
    assert (tmp_path / "errors/0000.json").is_file()


# --- schema validation -------------------------------------------------------

def test_validate_accepts_a_well_formed_reply():
    assert label_frames.validate(LABELS) == LABELS


@pytest.mark.parametrize("broken", [
    {"categories": [], "unresolved_regions": [], "extra": 1},          # extra top-level key
    {"categories": []},                                                # missing top-level key
    {"categories": [{"category": "table", "sam_prompt": "table"}], "unresolved_regions": []},
    {"categories": [], "unresolved_regions": [{"location": "left"}]},
])
def test_validate_rejects_wrong_shapes(broken):
    with pytest.raises(ValueError):
        label_frames.validate(broken)


def test_validate_rejects_bad_enum():
    bad = {"categories": [{**LABELS["categories"][0], "certainty": "maybe"}],
           "unresolved_regions": []}
    with pytest.raises(ValueError, match="Invalid enum"):
        label_frames.validate(bad)


@pytest.mark.parametrize("field", ["category", "sam_prompt"])
def test_validate_rejects_empty_category_or_prompt(field):
    bad = {"categories": [{**LABELS["categories"][0], field: "   "}], "unresolved_regions": []}
    with pytest.raises(ValueError, match="Empty category/prompt"):
        label_frames.validate(bad)


def test_validate_rejects_repeated_category():
    """SAM would be prompted twice for the same thing, doubling its masks."""
    repeated = {"categories": [LABELS["categories"][0], LABELS["categories"][0]],
                "unresolved_regions": []}
    with pytest.raises(ValueError, match="Repeated category"):
        label_frames.validate(repeated)


def test_validate_is_case_insensitive_about_repeats():
    second = {**LABELS["categories"][0], "category": "Table"}
    with pytest.raises(ValueError, match="Repeated category"):
        label_frames.validate({"categories": [LABELS["categories"][0], second],
                               "unresolved_regions": []})


# --- run configuration -------------------------------------------------------

def test_batch_records_carry_the_image_hash(tmp_path):
    frame = tmp_path / "frames/scene0000_00/frame_000011.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpeg-bytes")
    batch = tmp_path / "target_500.json"
    batch.write_text(json.dumps({"scene_id": "scene0000_00", "frame_indices": [11]}))

    scene_id, _, records = label_frames.build_records(batch, tmp_path / "frames")
    assert scene_id == "scene0000_00"
    assert records[0]["sequence_index"] == 0 and records[0]["source_frame_id"] == 11
    assert len(records[0]["source_sha256"]) == 64


def test_missing_frame_image_is_fatal(tmp_path):
    batch = tmp_path / "target_500.json"
    batch.write_text(json.dumps({"scene_id": "scene0000_00", "frame_indices": [11]}))
    with pytest.raises(SystemExit, match="Missing frame image"):
        label_frames.build_records(batch, tmp_path / "frames")


def test_gateway_credentials_come_from_the_api_env(monkeypatch, tmp_path):
    """The .env lives with the other gateway clients, not beside this module."""
    monkeypatch.setattr(label_frames.config, "GATEWAY_ENV_FILE", tmp_path / ".env")
    (tmp_path / ".env").write_text('export VIBEG_API_KEY="from-dotenv"\n# comment\nBAD_LINE\n')
    monkeypatch.delenv("VIBEG_API_KEY", raising=False)
    assert label_frames.settings()["VIBEG_API_KEY"] == "from-dotenv"


def test_environment_overrides_the_api_env(monkeypatch, tmp_path):
    monkeypatch.setattr(label_frames.config, "GATEWAY_ENV_FILE", tmp_path / ".env")
    (tmp_path / ".env").write_text("VIBEG_API_KEY=from-dotenv\n")
    monkeypatch.setenv("VIBEG_API_KEY", "from-env")
    assert label_frames.settings()["VIBEG_API_KEY"] == "from-env"


def test_prompt_ships_with_the_tree_and_is_recorded_by_hash():
    """The prompt is part of the request contract, so it is versioned with the code."""
    prompt = (label_frames.config.PROMPT_ROOT / "prompt_en.txt")
    assert prompt.is_file() and "JSON" in prompt.read_text()
