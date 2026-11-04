"""Build the ground-truth instance viewer.

The harness has been scored against nothing so far: every comparison between runs
argued about object counts with no reference to what the scene actually contains.
This page shows that reference - the ScanNet instance annotation for one scene -
so a claim such as "Object30 merged five pictures" can be checked against what is
really there rather than against another run's opinion.

Three files carry the annotation and all three are needed:

  *_vh_clean_2.ply          vertex positions and colours, 81k vertices
  *_vh_clean_2.0.010000.segs.json   one segment id per vertex, same order
  *.aggregation.json        segment id -> instance id -> label

The page draws points coloured per instance and lists the instances beside it.
It does not overlay a harness run: the two live in different coordinate frames
(ScanNet's original frame versus the Sim(3)-corrected AMB3R frame), and mixing
them before that is resolved would invite reading a misalignment as a detection
error.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from src import config

TEMPLATES = config.WEB_ROOT / "gt_instances"
DEFAULT_SCENE = Path("/home/seer/wzt/data/scannet_v2/scene0000_00")
DEFAULT_OUTPUT = config.SITE_ROOT / "gt_instances"

PLY_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint": "<u4", "int": "<i4", "short": "<i2"}


def read_vertex_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions and colours from a binary PLY, ignoring any element after `vertex`.

    The existing reader in `backproject_masks` rejects files that carry faces,
    which is right for the clouds this tree writes; the ScanNet meshes do carry
    them, and the faces are not needed to draw points.
    """
    fields, count, element = [], None, None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY: {path}")
        if handle.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError(f"Expected binary little-endian: {path}")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            tokens = line.decode("ascii").split()
            if tokens[:1] == ["element"]:
                element, size = tokens[1], int(tokens[2])
                if element == "vertex":
                    count = size
            elif tokens[:1] == ["property"] and element == "vertex":
                if tokens[1] == "list":
                    raise ValueError("Vertex list properties are unsupported")
                if tokens[1] not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY type {tokens[1]}")
                fields.append((tokens[2], PLY_TYPES[tokens[1]]))
            elif tokens == ["end_header"]:
                break
        if count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        vertices = np.fromfile(handle, dtype=np.dtype(fields), count=count)
    if len(vertices) != count:
        raise ValueError(f"Truncated PLY vertices: {path}")
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    colors = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.uint8)
    return points, colors


def read_instances(scene: Path, vertex_count: int):
    """One instance id per vertex, plus the instance table the page lists."""
    stem = scene.name
    segs_path = scene / f"{stem}_vh_clean_2.0.010000.segs.json"
    if not segs_path.is_file():
        segs_path = scene / f"{stem}_vh_clean_2.segs.json"
    segments = json.loads(segs_path.read_text())["segIndices"]
    if len(segments) != vertex_count:
        raise ValueError(f'{segs_path.name} lists {len(segments)} segments for {vertex_count} vertices')

    aggregation = json.loads((scene / f"{stem}.aggregation.json").read_text())
    # Segment ids are dense but not zero-based: build the vertex lookup through an
    # index array rather than assuming `segment == vertex` or a max-plus-one table.
    order = sorted(set(segments))
    position = {segment: index for index, segment in enumerate(order)}
    lookup = np.full(len(order), -1, dtype=np.int16)
    table = []
    for group in aggregation["segGroups"]:
        instance = int(group["id"])
        for segment in group["segments"]:
            if segment not in position:
                continue
            slot = position[segment]
            if lookup[slot] != -1:
                raise ValueError(f'Segment {segment} belongs to two instances')
            lookup[slot] = instance
        table.append({"id": instance, "label": group["label"], "segments": len(group["segments"])})
    per_vertex = lookup[np.searchsorted(order, segments)]
    return per_vertex.astype(np.int16), table


def encode(points: np.ndarray, instances: np.ndarray) -> dict:
    """Quantise positions to 16-bit offsets; instance ids are small enough for u8.

    The stride is exactly 7 bytes - three little-endian u16 then one u8 - so the
    page can decode with a fixed offset. Building it field by field rather than
    with column_stack is deliberate: stacking a u8 next to u16 fields promotes the
    whole row to uint16 and silently yields 8 bytes per vertex, which shifts every
    decoded point by one byte after the first.
    """
    low = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - low, 1e-6)
    quantised = np.round((points - low) / span * 65535).astype(np.uint16)
    # An unassigned vertex carries -1; u8 has no room for it, so instance 0 is
    # reserved as "no instance" and the table below starts at 1.
    codes = np.where(instances < 0, 0, instances + 1).astype(np.uint8)
    packed = np.empty((len(points), 7), dtype=np.uint8)
    for axis in range(3):
        packed[:, axis * 2] = quantised[:, axis] & 0xFF
        packed[:, axis * 2 + 1] = quantised[:, axis] >> 8
    packed[:, 6] = codes
    return {"minimum": low.astype(float).round(4).tolist(), "span": span.astype(float).round(4).tolist(),
            "count": int(len(points)), "payload": base64.b64encode(packed.tobytes()).decode("ascii")}


def build(scene: Path, output: Path) -> None:
    scene, output = scene.resolve(), output.resolve()
    points, colors = read_vertex_ply(scene / f"{scene.name}_vh_clean_2.ply")
    instances, table = read_instances(scene, len(points))

    counts = {int(instance): 0 for instance in instances if instance >= 0}
    for instance, total in zip(*np.unique(instances[instances >= 0], return_counts=True)):
        counts[int(instance)] = int(total)
    for row in table:
        row["points"] = counts.get(row["id"], 0)
    table.sort(key=lambda row: (-row["points"], row["label"]))

    output.mkdir(parents=True, exist_ok=True)
    payload = encode(points, instances)
    payload.update(scene=scene.name, instances=table,
                   unassigned=int((instances < 0).sum()), vertices=int(len(points)))
    (output / "gt.js").write_text(
        "window.GT_INSTANCES=" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + ";\n")

    for name in ("index.html", "page.css", "page.js"):
        (output / name).write_text((TEMPLATES / name).read_text())
    print(f"wrote {output} ({payload['count']} vertices, {len(table)} instances)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.scene, args.output)


if __name__ == "__main__":
    main()
