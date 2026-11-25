"""Build a self-contained page for reviewing physics and appearance estimates.

Reads generation output alongside physics_appearance.json and sizes.json,
embeds the mesh and key-frame images as base64 so the page works over file://.

Usage::

    .venv-sam3d/bin/python pagegen/build_physics_page.py output/scene0000_00/gt
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
TEMPLATES = ROOT / "pagegen" / "physics"
DEFAULT_OUTPUT = ROOT / "site" / "physics"

Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> tuple:
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


def encode_mesh(mesh: trimesh.Trimesh, target_faces: int, scale: float) -> dict:
    vertices, normals, faces, colours = decimate(mesh, target_faces)
    # Centre XY, sit on Z=0, apply scale
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    centre = (minimum + maximum) / 2
    vertices = vertices - np.array([centre[0], centre[1], minimum[2]])
    vertices = vertices * scale
    maximum = vertices.max(axis=0)
    minimum = vertices.min(axis=0)
    index_dtype = np.uint16 if len(vertices) < 65536 else np.uint32
    indices = faces.astype(index_dtype).ravel()

    def enc(a: np.ndarray) -> str:
        return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")

    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "positions": enc(vertices.astype(np.float32)),
        "normals": enc(normals.astype(np.float32)),
        "colours": enc(colours),
        "indices": enc(indices),
        "index_bits": 16 if index_dtype is np.uint16 else 32,
        "extents": np.ptp(vertices, axis=0).round(4).tolist(),
    }


def encode_image(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build(generation: Path, output: Path, target_faces: int) -> None:
    generation = generation.resolve()
    manifest = json.loads((generation / "manifest.json").read_text())
    handoff = Path(manifest["source_handoff"]["path"])
    scene = {o["object_id"]: o for o in json.loads((handoff / "objects.json").read_text())}
    records = [r for r in json.loads((generation / "objects.json").read_text())
               if r["status"] == "success"]

    pa_map = {o["object_id"]: o
              for o in json.loads((generation / "physics_appearance.json").read_text()).get("objects", [])}
    sz_map = {o["object_id"]: o
              for o in json.loads((generation / "sizes.json").read_text()).get("objects", [])}

    # Prefer assembly_handoff physics reports when available — they carry the
    # actual computed inertia / center-of-mass / kinematic fallback reason.
    handoff_dir = generation / "assembly_handoff"
    asm_objects: dict[str, dict] = {}
    if handoff_dir.is_dir():
        try:
            asm_objects = {o["object_id"]: o
                           for o in json.loads((handoff_dir / "objects.json").read_text())}
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    output.mkdir(parents=True, exist_ok=True)

    published = []
    for record in sorted(records, key=lambda r: r["object_id"]):
        oid = record["object_id"]
        pa = pa_map.get(oid, {})
        sz = sz_map.get(oid, {})
        src = scene.get(oid, {})
        asm = asm_objects.get(oid, {})

        # Read the actual physics report written by configure_usd_physics.py
        # so the page shows real inertia / center-of-mass / kinematic fallback.
        phys_report = {}
        phys_path = handoff_dir / "usds" / oid / "physics.physics.json"
        if phys_path.exists():
            try:
                phys_report = json.loads(phys_path.read_text())
            except json.JSONDecodeError:
                pass

        mesh = trimesh.load(generation / record["mesh"], force="mesh", process=False)
        scale = sz.get("scale_factor", 1.0)

        # Key-frame images (up to 3)
        keyframes = []
        for kf in src.get("key_frames", [])[:3]:
            rgb_path = handoff / kf["rgb"]
            if rgb_path.exists():
                keyframes.append({
                    "role": kf.get("role", ""),
                    "seq": kf.get("seq"),
                    "visible_pixels": kf.get("visible_pixels"),
                    "image": encode_image(rgb_path),
                })

        entry = {
            "object_id": oid,
            "label": record.get("label") or "",
            "mesh": encode_mesh(mesh, target_faces, scale),
            "physics": {
                "mass_kg": phys_report.get("mass_kg") or pa.get("mass_kg"),
                "roughness": pa.get("roughness"),
                "metallic": pa.get("metallic"),
                "opacity": 1.0,
                "body_type": phys_report.get("body_type") or pa.get("body_type"),
                "kinematic": phys_report.get("kinematic", pa.get("kinematic", False)),
                "kinematic_reason": asm.get("physics", {}).get("demotion_reason"),
                "rigid_body_enabled": phys_report.get("rigid_body_enabled",
                                                       not pa.get("kinematic", False)),
                "collision_enabled": phys_report.get("collision_enabled", True),
                "collision_approximation": phys_report.get("collision_approximation",
                                                            pa.get("collision_approximation")),
                "collision_prim": phys_report.get("collision_prim"),
                "visual_faces": phys_report.get("visual_faces"),
                "collision_faces": phys_report.get("collision_faces"),
                "max_convex_hulls": 128,
                "voxel_resolution": 500000,
                "static_friction": phys_report.get("static_friction", 0.6),
                "dynamic_friction": phys_report.get("dynamic_friction", 0.5),
                "restitution": phys_report.get("restitution", 0.0),
                "solver_velocity_iterations": phys_report.get("solver_velocity_iterations"),
                "solver_position_iterations": phys_report.get("solver_position_iterations"),
                "center_of_mass": phys_report.get("center_of_mass_m"),
                "diagonal_inertia": phys_report.get("diagonal_inertia_kg_m2"),
                "principal_axes": phys_report.get("principal_axes_wxyz"),
                "mass_distribution_assumption": phys_report.get("mass_distribution_assumption"),
                "convex_decomposition_settings": phys_report.get("convex_decomposition_settings"),
                "confidence": pa.get("confidence"),
            },
            "scale": {
                "scale_factor": sz.get("scale_factor"),
                "height_m": sz.get("height_m"),
                "source": sz.get("source"),
                "mesh_scene_z_thin_frac": sz.get("mesh_scene_z_thin_frac"),
                "footprint_long_m": round((sz.get("mesh_normalised") or {}).get("footprint_long", 0) * (sz.get("scale_factor") or 1), 4),
                "footprint_short_m": round((sz.get("mesh_normalised") or {}).get("footprint_short", 0) * (sz.get("scale_factor") or 1), 4),
                "on_floor": sz.get("on_floor"),
            },
            "key_frames": keyframes,
        }
        published.append(entry)
        print(f"{oid:>10}  {record.get('label') or '-':<20}  "
              f"mass={pa.get('mass_kg')}kg  conf={pa.get('confidence')}")

    # Counts summarising each pipeline stage, shown on the pipeline tab.
    size_list = list(sz_map.values())
    footprint_n = sum(1 for s in size_list if str(s.get("source", "")).startswith("scan footprint"))
    override_n = sum(1 for s in size_list if s.get("source") == "manual override")
    conf_counts = {}
    for e in published:
        c = e["physics"].get("confidence") or "?"
        conf_counts[c] = conf_counts.get(c, 0) + 1
    bt_counts: dict[str, int] = {}
    for e in published:
        bt = e["physics"].get("body_type") or "dynamic"
        bt_counts[bt] = bt_counts.get(bt, 0) + 1
    forced_n = sum(1 for e in published if e["physics"].get("kinematic_reason"))
    pipeline_stats = {
        "sizes": {
            "total": len(size_list),
            "z_height": len(size_list) - footprint_n - override_n,
            "footprint": footprint_n,
            "overrides": override_n,
        },
        "physics": {
            "total": len(published),
            "model": conf_counts.get("model", 0),
            "clamped": conf_counts.get("model+clamped", 0),
            "fallback": conf_counts.get("fallback", 0),
        },
        "handoff": {
            "total": len(published),
            "dynamic": bt_counts.get("dynamic", 0),
            "kinematic": bt_counts.get("kinematic", 0),
            "static": bt_counts.get("static", 0),
            "forced": forced_n,
            "built": handoff_dir.is_dir(),
        },
    }

    payload = {"generation": str(generation), "objects": published,
               "pipeline_stats": pipeline_stats}
    (output / "physics.js").write_text(
        "window.PHYSICS=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    for name in ("index.html", "page.css", "page.js"):
        shutil.copyfile(TEMPLATES / name, output / name)

    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(f"\nwrote {output}: {len(published)} objects, {total / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generation", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-faces", type=int, default=40000)
    args = parser.parse_args()
    build(args.generation, args.output, args.target_faces)


if __name__ == "__main__":
    main()
