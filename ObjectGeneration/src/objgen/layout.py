"""Shared per-image, per-run artifact layout (no model dependencies)."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import ROOT


STAGES = ("input", "sam3d", "usd", "physics")


ASSETS_ROOT = ROOT / "assets"


def object_output_parent(output_root, image):
    """Mirror asset categories; a multi-view directory is one object, not each angle."""
    image = Path(image).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    try:
        relative = image.relative_to(ASSETS_ROOT.resolve())
    except ValueError:
        return root / image.stem
    if len(relative.parts) >= 3:
        # assets/SKU/0/90.jpg -> output/SKU/0
        return root / relative.parent
    # assets/Beverages_And_snacks/1.jpg -> output/Beverages_And_snacks/1
    return root / relative.with_suffix("")


def create_run(output_root, image):
    """Allocate a unique run; never reuse a prior run's output directory."""
    parent = object_output_parent(output_root, image)
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        started = datetime.now(ZoneInfo("Asia/Shanghai"))
        run = parent / started.strftime("%Y%m%d_%H%M%S_%f")
        try:
            run.mkdir()
            break
        except FileExistsError:
            continue
    for stage in STAGES:
        (run / stage).mkdir()
    return run, started.isoformat()


def is_run(path):
    path = Path(path)
    return all((path / stage).is_dir() for stage in STAGES)


def resolve_mesh_input(path, output=None):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if not is_run(path):
            raise ValueError(f"Not a run directory: {path}")
        path = path / "sam3d/mesh.glb"
    if output is None:
        run = path.parent.parent
        if path.parent.name != "sam3d" or not is_run(run):
            raise ValueError("A legacy GLB requires --output; or pass a new run directory")
        output = run / "usd/object.usdc"
    return path, Path(output).expanduser().resolve()


def resolve_physics_input(path, output=None):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if not is_run(path):
            raise ValueError(f"Not a run directory: {path}")
        path = path / "usd/object.usdc"
    if output is None:
        run = path.parent.parent
        if path.parent.name == "usd" and is_run(run):
            output = run / "physics" / path.name
        else:
            output = path.with_name(path.stem + "_physics" + path.suffix)
    output = Path(output).expanduser().resolve()
    if output == path:
        raise ValueError("Physics output must differ from the visual USD input")
    return path, output
