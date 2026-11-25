"""Read a scene-understanding handoff and lay out per-object generation results.

The output mirrors the handoff it came from: one directory per scene and
handoff, one subdirectory per object and key-frame role, and two JSON files
that index everything.

    output/<scene_id>/<handoff dir name>/
      manifest.json                        what was generated from what
      objects.json                         one record per handoff object
      objects/<object_id>/<role>/sam3d/    mesh.glb splat.ply pose.json report.json
      objects/<object_id>/<role>/dualcraft/  product/ inputs/ report.json  (imported, optional)
      dualcraft.json                       which objects use an imported DualCraft asset

Directories are keyed by identity, never by timestamp, so the same object and
view always land in the same place. Each stage report records the sha256 of its
inputs and outputs; a rerun skips work whose inputs and outputs still match.

Nothing here imports a model, so selection, validation and resume decisions
can be tested and dry-run without a GPU.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from .config import ROOT
OUTPUT_ROOT = ROOT / "output"
SCHEMA = "assets/v1"
ROLES = ("primary", "secondary", "third")
STAGE = "sam3d"
OUTPUTS = ("mesh.glb", "splat.ply", "pose.json")
STAT_KEYS = ("vertices", "faces", "watertight", "seconds", "peak_vram_gib")
FRAMES = {
    "mesh": "object-local, Y-up, normalized so the longest side is about 1; no unit",
    "pose": "SAM3D layout estimate in the key-frame camera; rotation is usable, "
            "scale and translation are not metric",
}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def atomic_json(path, data) -> None:
    """Write through a temporary file so a crash never leaves half a report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_handoff(handoff) -> tuple[Path, dict, list]:
    handoff = Path(handoff).expanduser().resolve()
    manifest = json.loads((handoff / "manifest.json").read_text())
    objects = json.loads((handoff / "objects.json").read_text())
    if not str(manifest.get("schema_version", "")).startswith("handoff/"):
        raise ValueError(f"Not a handoff directory: {handoff}")
    if "scene_id" not in manifest:
        raise ValueError("manifest.json lacks scene_id")
    for record in objects:
        missing = [key for key in ("object_id", "key_frames") if key not in record]
        if missing:
            raise ValueError(f"objects.json record lacks {', '.join(missing)}: {record}")
    ids = [record["object_id"] for record in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("objects.json has duplicate object_id values")
    return handoff, manifest, objects


def default_output_root(handoff: Path, manifest: dict) -> Path:
    return OUTPUT_ROOT / manifest["scene_id"] / handoff.name


def stage_dir(root: Path, object_id: str, role: str) -> Path:
    return root / "objects" / object_id / role / STAGE


def inspect_inputs(rgb: Path, mask: Path) -> tuple[str | None, dict]:
    """Return (skip reason, facts) for one key frame; bad data never raises."""
    import numpy as np
    from PIL import Image
    for path in (rgb, mask):
        if not path.is_file():
            return f"missing file: {path.name}", {}
    try:
        with Image.open(rgb) as image:
            size = image.size
        with Image.open(mask) as picture:
            pixels = np.asarray(picture.convert("L")) > 0
    except OSError as exc:
        return f"unreadable image: {exc}", {}
    facts = {"width": size[0], "height": size[1], "mask_pixels": int(pixels.sum()),
             "rgb_sha256": sha256(rgb), "mask_sha256": sha256(mask)}
    if pixels.shape[::-1] != size:
        return f"rgb {size} and mask {pixels.shape[::-1]} sizes differ", facts
    if not facts["mask_pixels"]:
        return "mask is empty", facts
    return None, facts


def keyframe(record: dict, role: str) -> dict | None:
    for frame in record.get("key_frames") or []:
        if frame.get("role") == role:
            return frame
    return None


def plan(handoff: Path, objects: list, root: Path, role: str = "primary",
         only: set | None = None, include_structural: bool = False) -> list:
    """One task per handoff object. Selection marks tasks; it never drops one,
    so the index can always account for every object in the handoff."""
    if role not in ROLES:
        raise ValueError(f"view must be one of {ROLES}")
    if only:
        unknown = sorted(set(only) - {record["object_id"] for record in objects})
        if unknown:
            raise ValueError("Unknown object_id: " + ", ".join(unknown))
    tasks = []
    for record in objects:
        object_id = record["object_id"]
        task = {"object_id": object_id, "label": record.get("label"), "role": role,
                "selected": not only or object_id in only, "stage_dir": stage_dir(root, object_id, role)}
        frame = keyframe(record, role)
        if record.get("structural") and not include_structural:
            task["skip"] = "structural object"
        elif frame is None:
            task["skip"] = f"no {role} key frame"
        else:
            task["source"] = {"seq": frame.get("seq"), "rgb": frame["rgb"], "mask": frame["mask"]}
            task["rgb"], task["mask"] = handoff / frame["rgb"], handoff / frame["mask"]
        tasks.append(task)
    return tasks


def read_report(directory: Path) -> dict | None:
    try:
        return json.loads((Path(directory) / "report.json").read_text())
    except (OSError, ValueError):
        return None


def up_to_date(directory: Path, facts: dict, generator_key: dict) -> bool:
    """Reuse a finished stage only if its inputs, generator settings and every
    output file still match what its report recorded."""
    report = read_report(directory)
    if not report or report.get("status") != "success":
        return False
    recorded = report.get("inputs") or {}
    if any(recorded.get(key) != facts.get(key) for key in ("rgb_sha256", "mask_sha256")):
        return False
    if report.get("generator_key") != generator_key:
        return False
    outputs = report.get("outputs") or {}
    if set(outputs) != set(OUTPUTS):
        return False
    return all((Path(directory) / name).is_file() and sha256(Path(directory) / name) == digest
               for name, digest in outputs.items())


def dualcraft_registry(root: Path) -> dict:
    """object_id -> imported DualCraft asset (written by import_dualcraft.py)."""
    try:
        return json.loads((Path(root) / "dualcraft.json").read_text())
    except (OSError, ValueError):
        return {}


def dualcraft_record(root: Path, record: dict, entry: dict) -> dict:
    """An object whose asset came from DualCraft: the import report is the record."""
    directory = Path(root) / entry["stage"]
    report = read_report(directory)
    if report is None:
        return {**record, "status": "failed", "generator": "dualcraft",
                "error": f"{entry['stage']}/report.json is missing"}
    relative = Path(entry["stage"])
    return {**record, "view": entry.get("view", record["view"]), "status": report.get("status", "failed"),
            "generator": "dualcraft", "source": report.get("source"),
            "report": str(relative / "report.json"), "product": str(relative / "product"),
            "urdf": str(relative / "product" / "model.urdf"),
            "stats": {"links": report.get("links"), "joints": report.get("joints"),
                      "movable_joints": report.get("movable_joints")}}


def index_record(root: Path, task: dict) -> dict:
    record = {"object_id": task["object_id"], "label": task["label"], "view": task["role"]}
    entry = dualcraft_registry(root).get(task["object_id"])
    if entry is not None:
        return dualcraft_record(root, record, entry)
    if "skip" in task:
        return {**record, "status": "skipped", "reason": task["skip"]}
    report = read_report(task["stage_dir"])
    if report is None:
        return {**record, "status": "pending"}
    record.update(status=report.get("status", "failed"), generator=STAGE, source=report.get("source"))
    relative = task["stage_dir"].relative_to(root)
    record["report"] = str(relative / "report.json")
    if record["status"] == "success":
        record.update(mesh=str(relative / "mesh.glb"), splat=str(relative / "splat.ply"),
                      pose=str(relative / "pose.json"),
                      stats={key: report.get(key) for key in STAT_KEYS})
    elif report.get("reason") or report.get("error"):
        record["reason" if record["status"] == "skipped" else "error"] = report.get("reason") or report.get("error")
    return record


def write_index(root: Path, handoff: Path, manifest: dict, tasks: list, generator: dict) -> dict:
    """Rebuild objects.json and manifest.json from the per-object reports on disk.
    The reports are the record; the index can always be regenerated from them."""
    records = [index_record(root, task) for task in tasks]
    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    atomic_json(root / "objects.json", records)
    summary = {
        "schema_version": SCHEMA,
        "scene_id": manifest["scene_id"],
        "source_handoff": {
            "path": str(handoff), "schema_version": manifest.get("schema_version"),
            "producer": manifest.get("producer"),
            "manifest_sha256": sha256(handoff / "manifest.json"),
            "objects_sha256": sha256(handoff / "objects.json"),
        },
        "stage": STAGE,
        "view": tasks[0]["role"] if tasks else None,
        "frames": FRAMES,
        "generator": generator,
        "objects": len(records),
        "status_counts": counts,
        "generator_counts": generator_counts(records),
        "updated_at": now(),
    }
    atomic_json(root / "manifest.json", summary)
    return summary


def generator_counts(records: list) -> dict:
    counts = {}
    for record in records:
        if record.get("status") == "success":
            counts[record.get("generator", STAGE)] = counts.get(record.get("generator", STAGE), 0) + 1
    return counts


def reindex(root: Path) -> dict:
    """Rebuild objects.json after an import, keeping the generator settings the last
    SAM3D run recorded.  Tasks are re-planned from the handoff with that run's view."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    handoff, handoff_manifest, objects = load_handoff(manifest["source_handoff"]["path"])
    tasks = plan(handoff, objects, root, manifest.get("view") or "primary")
    return write_index(root, handoff, handoff_manifest, tasks, manifest.get("generator") or {})


def write_single_image(image, mask, label: str | None = None, scene_id: str | None = None,
                       handoffs_root: Path | None = None) -> Path:
    """Wrap one photo and its mask as a one-object handoff so it runs the same path.

    The scene id defaults to the image name plus a content hash: two different
    photos called front.jpg never share a directory, and rerunning the same
    photo lands in the same place, which is what makes resume work."""
    import numpy as np
    from PIL import Image
    image, mask = Path(image).expanduser().resolve(), Path(mask).expanduser().resolve()
    for path in (image, mask):
        if not path.is_file():
            raise FileNotFoundError(path)
    image_sha = sha256(image)
    scene_id = scene_id or f"{image.stem}_{image_sha[:8]}"
    handoff = (handoffs_root or OUTPUT_ROOT / ".handoffs") / scene_id / "single_image"
    frames = handoff / "keyframes" / "obj_0000"
    frames.mkdir(parents=True, exist_ok=True)
    rgb = frames / f"0000_rgb{image.suffix.lower()}"
    shutil.copyfile(image, rgb)
    with Image.open(mask) as picture:
        binary = (np.asarray(picture.convert("L")) > 0).astype(np.uint8) * 255
    Image.fromarray(binary).save(frames / "0000_mask.png")
    frame = {"seq": 0, "rgb": str(rgb.relative_to(handoff)), "mask": "keyframes/obj_0000/0000_mask.png",
             "role": "primary"}
    atomic_json(handoff / "objects.json", [{
        "object_id": "obj_0000", "label": label,
        "label_quality": "specific" if label else "generic", "structural": False,
        "key_frames": [frame],
    }])
    atomic_json(handoff / "manifest.json", {
        "schema_version": "handoff/v1", "scene_id": scene_id,
        "producer": {"name": "single_image", "image": str(image), "image_sha256": image_sha,
                     "mask": str(mask), "mask_sha256": sha256(mask)},
        "frames": 1, "objects": 1,
    })
    return handoff
