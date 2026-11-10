"""File operations shared by the two perception stages."""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from src import config


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def perception_dir(run):
    """A named experiment owns both labels and masks beneath its perception directory."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run):
        raise ValueError('Use an experiment name, not a path, for --run')
    root = config.ARTIFACTS_ROOT.resolve()
    target = root / 'experiments' / run / 'perception'
    if not target.resolve().is_relative_to(root):
        raise ValueError('Experiment output escapes the artifacts tree through a symlink')
    return target
