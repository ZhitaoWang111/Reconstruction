"""Focused contracts for the accepted-source gate and the geometry default."""
import json

import pytest

from src.harness import run as harness_run
from src.harness.observations import DEFAULT_SOURCE


def test_harness_default_source_points_at_the_amb3r_lifted_source():
    # The AMB3R lift is the only geometry this tree builds, so it is the default;
    # the retired cloud and mesh sources are no longer reachable at all.
    assert DEFAULT_SOURCE.parent.name == "scene0000_00"
    assert DEFAULT_SOURCE.name == "scene_amb3r_v1"


def test_harness_rejects_source_without_a_known_policy(tmp_path, monkeypatch, capsys):
    source = tmp_path / "legacy"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"frames": [{}]}))
    output = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "harness",
            "--source", str(source),
            "--output", str(output),
            "--frames", "1",
            "--window", "1",
            "--prepare-only",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        harness_run.main()

    assert exc.value.code == 2
    # The refusal has to name the one builder that produces an acceptable source,
    # or a mis-pointed --source is a dead end rather than a fixable mistake.
    assert "amb3r_mask_source" in capsys.readouterr().err
