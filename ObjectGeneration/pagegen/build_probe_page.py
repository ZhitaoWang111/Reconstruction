"""Build a self-contained page showing SAM3D objects generated from handoff key frames.

Reads one generation output directory written by ``scripts/run_generate.sh``:

    <generation>/manifest.json   names the source handoff
    <generation>/objects.json    one record per handoff object; successes are published

Key frames and scene facts (bounds, point count) are read from the handoff the
manifest points at, so nothing is copied twice.

Each mesh is decimated for the browser and embedded as base64 alongside its
generation report and its source key frame, so the page opens over ``file://``
with no server and no network.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shutil

import numpy as np
import open3d as o3d
import trimesh

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "pagegen" / "probe"
DEFAULT_OUTPUT = ROOT / "site" / "probe"

# mesh_to_usd.py rotates the GLB (Y-up) into the scene frame (Z-up). Applying it
# here means the page shows the object the way the USD will stand in the room.
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a mesh to roughly ``target_faces`` triangles, keeping vertex colours."""
    geometry = o3d.geometry.TriangleMesh()
    geometry.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    geometry.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    colours = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0
    geometry.vertex_colors = o3d.utility.Vector3dVector(colours)
    if len(mesh.faces) > target_faces:
        geometry = geometry.simplify_quadric_decimation(target_faces)
    geometry.compute_vertex_normals()
    vertices = np.asarray(geometry.vertices, dtype=np.float64) @ Y_UP_TO_Z_UP.T
    normals = np.asarray(geometry.vertex_normals, dtype=np.float64) @ Y_UP_TO_Z_UP.T
    faces = np.asarray(geometry.triangles, dtype=np.uint32)
    vertex_colours = (np.asarray(geometry.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
    return vertices, normals, faces, vertex_colours


def encode_mesh(mesh: trimesh.Trimesh, target_faces: int) -> dict:
    vertices, normals, faces, colours = decimate(mesh, target_faces)
    # Centre in XY and sit on Z=0, matching what mesh_to_usd.py does before writing
    # the USDA. Without it every object floats around its own arbitrary origin.
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    centre = (minimum + maximum) / 2
    vertices = vertices - np.array([centre[0], centre[1], minimum[2]])
    maximum = vertices.max(axis=0)
    minimum = vertices.min(axis=0)

    index_dtype = np.uint16 if len(vertices) < 65536 else np.uint32
    indices = faces.astype(index_dtype).ravel()

    def encode(array: np.ndarray) -> str:
        return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")

    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "positions": encode(vertices.astype(np.float32)),
        "normals": encode(normals.astype(np.float32)),
        "colours": encode(colours),
        "indices": encode(indices),
        "index_bits": 16 if index_dtype is np.uint16 else 32,
        "extents": np.ptp(vertices, axis=0).round(4).tolist(),
        "minimum": minimum.round(4).tolist(),
        "maximum": maximum.round(4).tolist(),
    }


def encode_cloud(path: Path, max_points: int) -> dict:
    """The object's scan points, shifted so the origin sits under its bottom centre.

    The scan frame is Z-up in metres, so only a translation is applied: the axes
    drawn in the page are the scan's own axes, and `origin_scan_m` says where
    that local origin lies in the scan.
    """
    cloud = trimesh.load(path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float64)
    colours = np.asarray(cloud.colors, dtype=np.uint8)[:, :3] if len(getattr(cloud, "colors", [])) else \
        np.full((len(points), 3), 180, dtype=np.uint8)
    total = len(points)
    if total > max_points:
        keep = np.random.default_rng(0).choice(total, max_points, replace=False)
        points, colours = points[keep], colours[keep]
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    origin = np.array([(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]])
    local = (points - origin).astype(np.float32)

    def encode(array: np.ndarray) -> str:
        return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")

    return {
        "point_count": int(len(local)),
        "point_count_total": int(total),
        "positions": encode(local),
        "colours": encode(colours),
        "extents": np.ptp(local, axis=0).round(4).tolist(),
        "origin_scan_m": origin.round(4).tolist(),
    }


def build(generation: Path, output: Path, target_faces: int, object_ids: list[str] | None,
          max_cloud_points: int = 60000) -> None:
    generation = generation.resolve()
    manifest = json.loads((generation / "manifest.json").read_text())
    handoff = Path(manifest["source_handoff"]["path"])
    scene = {record["object_id"]: record for record in json.loads((handoff / "objects.json").read_text())}
    records = [record for record in json.loads((generation / "objects.json").read_text())
               if record["status"] == "success"]
    if object_ids:
        unknown = sorted(set(object_ids) - {record["object_id"] for record in records})
        if unknown:
            raise SystemExit("No successful generation for: " + ", ".join(unknown))
        records = [record for record in records if record["object_id"] in object_ids]

    # Load sizes.json if present so each object entry carries its scale and height.
    sizes_path = generation / "sizes.json"
    sizes: dict[str, dict] = {}
    if sizes_path.is_file():
        for entry in json.loads(sizes_path.read_text()).get("objects", []):
            sizes[entry["object_id"]] = entry

    output.mkdir(parents=True, exist_ok=True)
    media = output / "media"
    if media.exists():
        shutil.rmtree(media)
    published = []
    for record in sorted(records, key=lambda item: item["object_id"]):
        object_id = record["object_id"]
        report_path = generation / record["report"]
        report = json.loads(report_path.read_text())
        source = scene[object_id]
        frame = next(item for item in source["key_frames"] if item["role"] == record["view"])

        media_dir = media / object_id
        media_dir.mkdir(parents=True, exist_ok=True)
        images = {}
        for kind in ("rgb", "mask"):
            origin = handoff / frame[kind]
            target = media_dir / f"{record['view']}_{kind}{origin.suffix}"
            shutil.copyfile(origin, target)
            images[kind] = str(target.relative_to(output))

        mesh = trimesh.load(generation / record["mesh"], force="mesh", process=False)
        size_entry = sizes.get(object_id, {})
        entry = {
            "object_id": object_id,
            "label": record["label"],
            "seq": frame["seq"],
            "source_frame_id": frame.get("source_frame_id"),
            "visible_pixels": frame.get("visible_pixels"),
            "handoff_bounds": source.get("bounds"),
            "point_count": source.get("point_count"),
            "images": images,
            "run_dir": str(report_path.parent.relative_to(generation)),
            "report": {"elapsed_seconds": report.get("seconds"), "seed": report["generator_key"]["seed"],
                       **{key: report.get(key) for key in ("vertices", "faces", "watertight", "peak_vram_gib")}},
            "size": {
                "scale_factor": size_entry.get("scale_factor"),
                "height_m": size_entry.get("height_m"),
                "source": size_entry.get("source"),
                "needs_review": size_entry.get("needs_review"),
                "flags": size_entry.get("flags", []),
                "override_note": size_entry.get("override_note"),
                "candidates_m": size_entry.get("candidates_m"),
            } if size_entry else None,
            "mesh": encode_mesh(mesh, target_faces),
            "cloud": encode_cloud(handoff / source["cloud"], max_cloud_points),
        }
        published.append(entry)
        size_mb = sum(len(entry["mesh"][key]) for key in
                      ("positions", "normals", "colours", "indices")) / 1e6
        scale_str = f"  scale={size_entry['scale_factor']:.3f}" if size_entry.get("scale_factor") else ""
        print(f"{object_id:>10} {record['label'] or '-':<14} "
              f"{entry['mesh']['face_count']:>6} tris {size_mb:5.2f} MB embedded{scale_str}")

    payload = {"generation": str(generation), "handoff": str(handoff), "objects": published}
    (output / "probe.js").write_text("window.PROBE=" + json.dumps(payload, separators=(",", ":")) + ";\n")
    for name in ("index.html", "page.css", "page.js"):
        shutil.copyfile(TEMPLATES / name, output / name)
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"wrote {output}: {len(published)} objects, {total / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generation", type=Path, help="Generation output directory, e.g. output/scene0000_00/gt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-faces", type=int, default=60000)
    parser.add_argument("--object-id", action="append", help="Repeat to publish a subset")
    parser.add_argument("--max-points", type=int, default=60000,
                        help="Subsample each scan cloud to at most this many points")
    args = parser.parse_args()
    build(args.generation, args.output, args.target_faces, args.object_id, args.max_points)


if __name__ == "__main__":
    main()
