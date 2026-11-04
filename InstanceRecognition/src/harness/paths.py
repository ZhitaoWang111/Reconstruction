"""Resolve paths recorded inside a run's config.json.

A run's `config.json` stores where its point-cloud source lives. Runs created
before the artifacts/site split stored that as a repo-relative string
(`runtime/mask_pointclouds/...`) and relied on the process working directory
happening to be the old tree root.

Reading such a run from this tree must not silently reinterpret the relative
value against the new root, so resolution is anchored to directories rather than
to the working directory. This lives in its own module — not in `run.py` — because
both the pipeline and the page builders need it, and `run.py` already depends on
the page builders.
"""
from pathlib import Path


def resolve_recorded_path(value, run_dir):
    """Resolve a config-recorded path against the artifacts tree that owns the run.

    Absolute values pass through. A relative value is resolved against the
    artifacts root above `run_dir`, which is where this tree writes sources.

    A value that already starts with the artifacts directory name was recorded
    relative to the repository root instead -- `--source
    artifacts/sources/...` given on the command line is stored verbatim. Anchoring
    that against the artifacts root doubled the prefix
    (`artifacts/artifacts/sources/...`), which only surfaced at publish time,
    after the whole run had been paid for. The distinction is visible in the value
    itself, so it is read there rather than guessed.

    This used to fall back to the pre-refactor tree's runtime root, because old
    runs recorded paths relative to it. That tree was deleted on 2026-10-19, so
    the fallback could only ever return a directory that is not there; what such a
    run needs is an error naming the recorded value, not a second guess.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    run_dir = Path(run_dir).expanduser().resolve()
    if path.parts and path.parts[0] == 'artifacts':
        return run_dir.parent.parent.parent / path
    return run_dir.parent.parent / path
