"""M1 input, revision, locking and resumable-stage contracts (no model imports)."""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SCHEMA_VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_config(config):
    if not isinstance(config, dict) or config.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported configuration schema_version')
    for name in ('model', 'base_url', 'api_env_file', 'blender'):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f'Invalid {name}')
    for name in ('request_timeout_seconds', 'max_completion_tokens', 'max_iterations', 'mcp_port', 'texture_size'):
        v = config.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'Invalid {name}')
    if not isinstance(config['mcp_port'], int) or not 1024 <= config['mcp_port'] <= 65535:
        raise ValueError('mcp_port out of range')
    if config['texture_size'] not in (1024, 2048, 4096):
        raise ValueError('texture_size must be 1024, 2048 or 4096')
    for name,default in [('max_api_calls',4),('max_total_tokens',60000),('max_wall_seconds',1200)]:
        value=config.get(name,default)
        if isinstance(value,bool) or not isinstance(value,int) or value<=0: raise ValueError('Invalid '+name)
        config[name]=value
    if not isinstance(config['max_iterations'],int): raise ValueError('max_iterations must be integer')
    if config.get('review_required') is not True:
        raise ValueError('Human review must be enabled for the first release')
    return config


def resolve_inputs(run):
    run = Path(run).resolve()
    if not all((run / p).is_dir() for p in ('input', 'sam3d', 'usd', 'physics')):
        raise ValueError('Expected a four-stage run directory')
    meta = json.loads((run / 'input/input.json').read_text())
    generation = json.loads((run / 'sam3d/report.json').read_text())

    def local_path(raw):
        path = Path(raw)
        path = (path if path.is_absolute() else run / path).resolve()
        if not path.is_relative_to(run):
            raise ValueError(f'Input outside run snapshot: {path}')
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    # Only use captured inputs, never fall back to mutable original source paths.
    image = local_path(meta.get('image') or generation['image'])
    mask = local_path(meta.get('mask') or generation['mask'])
    for key, path in (('image', image), ('mask', mask)):
        expected = meta.get(key + '_sha256')
        if expected and expected != digest(path):
            raise ValueError(f'{key} snapshot hash mismatch')
    files = {'image': image, 'mask': mask, 'glb': run / 'sam3d/mesh.glb',
             'usd': run / 'usd/object.usda', 'input_manifest': run / 'input/input.json',
             'generation_report': run / 'sam3d/report.json'}
    for name in ('object.physics.json', 'object.usda'):
        candidate = run / 'physics' / name
        if candidate.is_file():
            files['physics_' + name] = candidate
    if generation.get('status') != 'success':
        raise ValueError('SAM3D generation report is not successful')
    return {key: {'path': str(path), 'sha256': digest(path)} for key, path in files.items()}


@contextmanager
def run_lock(run):
    folder = Path(run) / 'usd/appearance'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.run.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another appearance task owns this run') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def allocate_revision(run):
    parent = Path(run) / 'usd/appearance'
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / ('v_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    path.mkdir()
    return path


def valid_stage(record, revision):
    if record.get('status') != 'success' or not record.get('outputs'):
        return False
    for relative, expected in record['outputs'].items():
        path = (Path(revision) / relative).resolve()
        if not path.is_relative_to(Path(revision).resolve()) or not path.is_file() or digest(path) != expected:
            return False
    return True
