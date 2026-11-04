"""Recorded source paths resolve against the tree they were written from."""
from pathlib import Path
from src.harness.paths import resolve_recorded_path

RUN = Path('/repo/artifacts/experiments/harness-v6-amb3r-150frames-20261020-1750')


def test_absolute_passes_through():
    value = '/elsewhere/scene'
    assert resolve_recorded_path(value, RUN) == Path(value)


def test_repo_relative_value_is_not_nested_under_artifacts_again():
    # `--source artifacts/sources/x` is stored verbatim. Anchoring it against the
    # artifacts root produced artifacts/artifacts/sources/x, which only failed at
    # publish time after a full run had been paid for.
    resolved = resolve_recorded_path('artifacts/sources/sceneX', RUN)
    assert resolved == Path('/repo/artifacts/sources/sceneX')
    assert resolved.parts.count('artifacts') == 1


def test_artifacts_relative_value_anchors_at_the_artifacts_root():
    # The other recorded form, as written by an older invocation.
    assert resolve_recorded_path('sources/sceneX', RUN) == Path('/repo/artifacts/sources/sceneX')
