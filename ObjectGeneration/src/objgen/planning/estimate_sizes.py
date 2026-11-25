"""Derive a metric scale factor for every generated object.

SAM3D meshes come out normalised (longest side ~1), so each needs one uniform scale
factor to place it at real-world size.  This script derives that factor from the
handoff's per-object scan point cloud.

Strategy
--------
1. **Z-height** (default): ``scan_Z_height / mesh_GLB_Y``.  Reliable when the scan
   cloud covers the full object height and the mesh is a solid 3-D shape.

2. **Footprint fallback**: ``scan_footprint_long / mesh_footprint_long`` (normalised
   back to height units so the output is still a height-based scale).  Used
   automatically when either reliability check fails:

   - *Flat-slab mesh*: after Y-up→Z-up rotation, mesh scene-Z / mesh-max < 0.05.
     SAM3D produced a surface tile; its Y extent is slab thickness, not height.
   - *Z/footprint mismatch > 3×*: the two independent estimates contradict strongly.

Extents use robust 1st–99th percentiles so stray points don't inflate them.

Manual overrides
----------------
Place ``sizes_overrides.json`` in the generation directory to pin specific objects
(e.g. tape-measured values).  Applied after the automatic pass.

Format::

    {
      "obj_0054": {"height_m": 2.05, "note": "door, tape-measured on site"},
      "obj_0042": {"scale_factor": 1.83}
    }

Usage::

    PYTHONPATH=src .venv-sam3d/bin/python -m objgen.planning.estimate_sizes output/scene0000_00/gt

Writes <generation>/sizes.json and prints a table. Nothing else is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
import trimesh

from ..config import ROOT
from ..dualcraft import dualcraft_product as dp  # noqa: E402

PLANK_RATIO = 0.1      # cloud thinnest/longest < this → planar-partial (diagnostic only)
FLAT_SLAB_RATIO = 0.05  # mesh scene-Z/max < this → SAM3D produced a surface tile
SF_Z_FP_MISMATCH = 3.0  # sf_z/sf_fp outside [1/N, N] → Z-height estimate unreliable

# Size confidence.  The scan only covers what the camera saw, so a scanned extent
# is a lower bound: an asset larger than the scan is unverified, an asset smaller
# than the scan contradicts the observation.
AXIS_AGREE = 1.25       # asset/scan within [1/N, N] on an axis counts as agreement
AXIS_CONTRADICT = 0.5   # asset/scan below this on any axis → low confidence
MIN_SCAN_POINTS = 100   # fewer points → percentile extents are noisy
FLOOR_GAP_M = (0.1, 0.3)  # scan bottom this far above the floor → base likely unscanned


def scan_extents(points: np.ndarray) -> dict:
    """Height along Z (1–99 pct), footprint along cloud's own principal XY axes,
    and the full 3-D PCA spans (longest→shortest) for cross-checking."""
    low, high = np.percentile(points[:, 2], [1, 99])
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    _, _, axes_xy = np.linalg.svd(xy, full_matrices=False)
    spans_xy = sorted(
        (float(np.ptp(np.percentile(xy @ axis, [1, 99]))) for axis in axes_xy), reverse=True
    )
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vh.T
    pca3 = list(
        float(x) for x in sorted(np.ptp(np.percentile(proj, [1, 99], axis=0), axis=0), reverse=True)
    )
    return {"height": float(high - low), "bottom": float(low), "top": float(high),
            "footprint_long": spans_xy[0], "footprint_short": spans_xy[1],
            "pca3": pca3,
            "planar_partial": bool(pca3[2] / pca3[0] < PLANK_RATIO if pca3[0] > 0 else False)}


def mesh_extents(path: Path) -> dict:
    """GLB is Y-up: Y is height, X and Z are the footprint.

    Also computes ``scene_z_thin_frac``: after applying the Y-up→Z-up rotation used
    in ``build_probe_page.py`` (GLB-Y→scene-Z), this is scene-Z / scene-max.  A value
    below FLAT_SLAB_RATIO means SAM3D produced a surface tile rather than a solid object.
    """
    mesh = trimesh.load(path, force="mesh", process=False)
    extent = np.ptp(mesh.vertices, axis=0)  # GLB X, Y, Z
    horizontal = sorted((float(extent[0]), float(extent[2])), reverse=True)
    # After Y_UP_TO_Z_UP: GLB-X→scene-X, GLB-Y→scene-Z, GLB-Z→scene-Y
    scene_z = float(extent[1])   # GLB Y becomes scene Z (vertical)
    scene_max = max(float(extent[0]), float(extent[1]), float(extent[2]))
    return {
        "height": float(extent[1]),
        "footprint_long": horizontal[0],
        "footprint_short": horizontal[1],
        "scene_z_thin_frac": round(scene_z / scene_max, 4) if scene_max > 0 else 1.0,
    }


def assess(label: str, scan: dict, mesh: dict) -> dict:
    """Choose a uniform scale factor: Z-height by default, footprint if Z is unreliable.

    Primary estimate: scan Z-height / mesh Y-height (same axis on both sides).

    Reliability checks — if either fires, fall back to footprint matching:
      1. flat-slab mesh: mesh scene-Z / mesh-max < FLAT_SLAB_RATIO.
         SAM3D produced a surface tile; mesh Y is slab thickness, not object height.
      2. sf_z/sf_footprint mismatch > SF_Z_FP_MISMATCH.
         The two independent estimates contradict each other strongly.

    Cross-check (PCA secondary) is kept for diagnostics only.
    """
    from_height = scan["height"]
    sf_fp_long = scan["footprint_long"] / mesh["footprint_long"] if mesh["footprint_long"] > 0 else 0
    from_footprint = sf_fp_long * mesh["height"]

    # PCA cross-check for diagnostics.
    pca3 = scan.get("pca3", [from_height, scan["footprint_long"], scan["footprint_short"]])
    mesh_spans = sorted(
        [mesh["height"], mesh["footprint_long"], mesh["footprint_short"]], reverse=True
    )
    pca_candidates = [span / mspan * mesh["height"]
                      for span, mspan in zip(pca3, mesh_spans) if mspan > 0]
    secondary = min(pca_candidates, key=lambda x: abs(x - from_height)) if pca_candidates else from_footprint
    ratio = from_height / secondary if secondary > 0 else 1.0

    # Reliability checks: does Z-height give a trustworthy scale?
    sf_z_raw = scan["height"] / mesh["height"] if mesh["height"] > 0 else 0
    sf_zfp_ratio = sf_z_raw / sf_fp_long if sf_fp_long > 0 else 1.0
    flat_slab = mesh.get("scene_z_thin_frac", 1.0) < FLAT_SLAB_RATIO
    zfp_mismatch = sf_zfp_ratio > SF_Z_FP_MISMATCH or sf_zfp_ratio < 1.0 / SF_Z_FP_MISMATCH
    z_unreliable = flat_slab or zfp_mismatch

    if z_unreliable:
        height = from_footprint
        reason_parts = []
        if flat_slab:
            reason_parts.append(f"flat-slab mesh (scene Z/max={mesh['scene_z_thin_frac']:.3f})")
        if zfp_mismatch:
            reason_parts.append(f"sf_z/sf_fp={sf_zfp_ratio:.1f}x")
        source = "scan footprint (" + ", ".join(reason_parts) + ")"
    else:
        height = from_height
        source = "scan height"

    return {"height_m": round(height, 3), "source": source,
            "candidates_m": {"scan_height": round(from_height, 3),
                             "footprint": round(from_footprint, 3),
                             "pca_secondary": round(secondary, 3)},
            "proportion_ratio": round(ratio, 3), "label_range_m": None,
            "needs_review": False, "flags": []}


def dualcraft_entry(generation: Path, record: dict, label: str, scan: dict, points: int,
                    floor_z: float | None) -> dict:
    """A DualCraft asset is already metric (built to the size it was given), so the
    scale is 1; what is left is to check it against the scan the same way."""
    product = dp.read_product(generation / record["product"])
    bounds = dp.rest_bounds(product)
    extent = bounds[1] - bounds[0]
    footprint = sorted((float(extent[0]), float(extent[1])), reverse=True)
    entry = {"object_id": record["object_id"], "label": label, "points": points,
             "generator": "dualcraft", "height_m": round(float(extent[2]), 3),
             "source": "dualcraft (metric, built to the given size)",
             "candidates_m": {"scan_height": round(scan["height"], 3)},
             "proportion_ratio": None, "label_range_m": None, "needs_review": False, "flags": [],
             "scan_m": {k: round(v, 3) for k, v in scan.items() if k != "pca3" and not isinstance(v, bool)},
             "scan_planar_partial": scan.get("planar_partial", False),
             # mesh_normalised x scale_factor is the asset extent; with scale 1 it is metric.
             "mesh_normalised": {"height": round(float(extent[2]), 3),
                                 "footprint_long": round(footprint[0], 3),
                                 "footprint_short": round(footprint[1], 3)},
             "mesh_scene_z_thin_frac": None, "scale_factor": 1.0}
    if floor_z is not None:
        entry["on_floor"] = abs(scan["bottom"] - floor_z) < 0.1
    given = json.loads((generation / record["product"]).parent.joinpath(
        "inputs", "scannet_handoff.json").read_text()).get("size") or {}
    if given:
        tolerance = 0.10
        dims = {"width": float(extent[1]), "depth": float(extent[0]), "height": float(extent[2])}
        entry["given_size_m"] = {k: given.get(k) for k in ("width", "depth", "height", "measured")}
        entry["given_ratio"] = {k: round(dims[k] / given[k], 3) for k in dims if given.get(k)}
        outside = [f"{k} {r:.2f}x" for k, r in entry["given_ratio"].items() if abs(r - 1) > tolerance]
        if outside:
            entry["flags"].append("asset outside the given size by more than 10%: " + ", ".join(outside))
    return entry


def size_confidence(entry: dict, floor_z: float | None) -> None:
    """Grade the chosen scale against the scan; sets confidence, flags, axis_ratio.

    high   — Z height used, every axis agrees with the scan, scan dense enough.
    medium — no contradiction, but something is unverified (partial or sparse
             scan, asset larger than the scan on an axis, base possibly unscanned)
             or an axis is moderately smaller than the scan.
    low    — Z height unusable (footprint fallback), or the asset is less than
             half the scanned extent on some axis.
    """
    if entry["source"] == "manual override":
        entry.update(confidence="high", flags=["manual override"], needs_review=False)
        return
    sf, mesh, scan = entry["scale_factor"], entry["mesh_normalised"], entry["scan_m"]
    ratio = {axis: round(mesh[axis] * sf / scan[axis], 3) if scan[axis] > 0 else None
             for axis in ("height", "footprint_long", "footprint_short")}
    low, flags = False, list(entry.get("flags", []))
    if entry["source"].startswith("scan footprint"):
        low = True
        flags.append("Z height unusable, scale set from footprint: " + entry["source"])
    for axis, r in ratio.items():
        if r is None:
            continue
        if r < 1 / AXIS_AGREE:
            low = low or r < AXIS_CONTRADICT
            flags.append(f"asset {axis} is {r:.2f}x the scanned extent")
        elif r > AXIS_AGREE:
            flags.append(f"asset {axis} is {r:.2f}x the scanned extent (scan may be partial)")
    if entry["points"] < MIN_SCAN_POINTS:
        flags.append(f"sparse scan ({entry['points']} points)")
    if floor_z is not None and FLOOR_GAP_M[0] < scan["bottom"] - floor_z < FLOOR_GAP_M[1]:
        flags.append(f"scan bottom {scan['bottom'] - floor_z:.2f} m above floor; if it stands "
                     f"on the floor its base is unscanned and the size is underestimated")
    entry.update(axis_ratio=ratio, flags=flags, needs_review=low,
                 confidence="low" if low else "medium" if flags else "high")


def apply_overrides(results: list[dict], overrides: dict[str, dict], mesh_index: dict[str, dict]) -> None:
    """Replace auto-computed size for objects listed in sizes_overrides.json.

    ``height_m`` is the primary field when both are given — it is what someone
    with a tape measure actually records. ``scale_factor`` is derived from it
    against the mesh Y extent, unless the override supplies ``scale_factor``
    and no ``height_m``, in which case height is derived from the scale.
    Mutates entries in *results* in place.
    """
    index = {r["object_id"]: r for r in results}
    for oid, ov in overrides.items():
        if oid not in index:
            print(f"  WARNING: override for {oid!r} not in results (skipped)")
            continue
        if "height_m" not in ov and "scale_factor" not in ov:
            print(f"  WARNING: override for {oid!r} has neither height_m nor scale_factor (skipped)")
            continue
        entry = index[oid]
        m = mesh_index[oid]
        if "height_m" in ov:
            height = float(ov["height_m"])
            sf = height / m["height"] if m["height"] > 0 else 0.0
        else:
            sf = float(ov["scale_factor"])
            height = sf * m["height"]
        entry["scale_factor"] = round(sf, 4)
        entry["height_m"] = round(height, 3)
        entry["source"] = "manual override"
        if "note" in ov:
            entry["override_note"] = ov["note"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generation", type=Path, help="Generation output directory (has manifest.json)")
    args = parser.parse_args()
    generation = args.generation.resolve()
    manifest = json.loads((generation / "manifest.json").read_text())
    handoff = Path(manifest["source_handoff"]["path"])
    scene = {r["object_id"]: r for r in json.loads((handoff / "objects.json").read_text())}
    floor = [r for r in scene.values() if r.get("label") == "floor"]
    floor_z = None
    if floor:
        cloud = trimesh.load(handoff / floor[0]["cloud"])
        floor_z = float(np.median(np.asarray(cloud.vertices)[:, 2]))

    overrides_path = generation / "sizes_overrides.json"
    overrides: dict[str, dict] = {}
    if overrides_path.is_file():
        overrides = json.loads(overrides_path.read_text())
        print(f"Loaded {len(overrides)} manual override(s) from {overrides_path.name}")

    results = []
    mesh_index: dict[str, dict] = {}
    for record in json.loads((generation / "objects.json").read_text()):
        if record["status"] != "success":
            continue
        oid = record["object_id"]
        source = scene[oid]
        points = np.asarray(trimesh.load(handoff / source["cloud"]).vertices)
        scan = scan_extents(points)
        if record.get("generator") == "dualcraft":
            entry = dualcraft_entry(generation, record, record["label"] or "", scan, len(points), floor_z)
            mesh_index[oid] = entry["mesh_normalised"]
            results.append(entry)
            continue
        mesh = mesh_extents(generation / record["mesh"])
        mesh_index[oid] = mesh
        entry = {"object_id": oid, "label": record["label"], "points": len(points), "generator": "sam3d",
                 **assess(record["label"] or "", scan, mesh),
                 "scan_m": {k: round(v, 3) for k, v in scan.items() if k != "pca3" and not isinstance(v, bool)},
                 "scan_planar_partial": scan.get("planar_partial", False),
                 "mesh_normalised": {k: round(v, 3) for k, v in mesh.items() if k != "scene_z_thin_frac"},
                 "mesh_scene_z_thin_frac": mesh.get("scene_z_thin_frac")}
        if floor_z is not None:
            entry["on_floor"] = abs(scan["bottom"] - floor_z) < 0.1
        entry["scale_factor"] = round(entry["height_m"] / mesh["height"], 4)
        results.append(entry)

    if overrides:
        apply_overrides(results, overrides, mesh_index)
    for entry in results:
        size_confidence(entry, floor_z)

    out = {"schema_version": "sizes/v1", "generation": str(generation), "floor_z_m": floor_z,
           "method": "scan Z-height cross-checked against 3-D PCA and a per-label height prior; "
                     "manual overrides from sizes_overrides.json take precedence",
           "objects": results,
           "overrides_applied": sorted(overrides.keys()) if overrides else []}
    (generation / "sizes.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    print(f"{'object':9} {'label':16} {'height':>7} {'scan_h':>7} {'by_foot':>7} {'pca_2nd':>7} {'ratio':>6}  source")
    for r in sorted(results, key=lambda r: r["label"] or ""):
        c = r.get("candidates_m", {})
        src = r.get("source", "")
        ovr = (" [override]" if src == "manual override" else " [footprint]" if src.startswith("scan footprint")
               else " [dualcraft]" if r.get("generator") == "dualcraft" else "")
        print(f"{r['object_id']:9} {(r['label'] or '-'):16} {r['height_m']:7.2f} {c.get('scan_height', 0):7.2f} "
              f"{c.get('footprint') or 0:7.2f} {c.get('pca_secondary') or 0:7.2f} {r.get('proportion_ratio') or 0:6.2f}{ovr}")
    fp_count = sum(1 for r in results if r.get("source", "").startswith("scan footprint"))
    ovr_count = len(overrides)
    dc_count = sum(1 for r in results if r.get("generator") == "dualcraft")
    print(f"\n{len(results)} objects: {len(results)-fp_count-ovr_count-dc_count} z-height, {fp_count} footprint fallback, "
          f"{ovr_count} manual override, {dc_count} dualcraft (scale 1) -> {generation / 'sizes.json'}")


if __name__ == "__main__":
    main()
