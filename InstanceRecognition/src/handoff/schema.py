"""The handoff contract between scene understanding and the object/scene modules.

Two producers must be able to write this format: the ScanNet ground truth today,
and the harness once its results are trusted. Everything here is therefore shaped
by one rule - a field is required only if both producers can fill it. Anything
that only the GT can answer (a camera pose, a triangle mesh) is optional, so the
contract does not have to be revised on the day the harness starts writing.

The schema is code rather than prose because three separate readers depend on it:
`from_gt.py` writes it, `from_harness.py` will write it, and `check.py` verifies
whatever either one produced. If the field names lived only in a document, the
two producers would drift and `check.py` would be checking the wrong thing.

Coordinate frame
----------------
Points and poses are in `amb3r_floor` (docs/amb3r-floor-frame.md): AMB3R's own
reconstruction, scaled and stood upright on the floor we segmented - metres, +Z up,
z = 0 on the floor, origin below the first camera. No ScanNet axisAlignment, no GT pose.
The manifest's `frame` block carries the AMB3R -> frame matrix and where the scale came
from (`scale.source`: `oracle_gt_poses` on ScanNet until a GT-free scale exists). The GT
producer maps the ScanNet mesh and poses into the same frame, so GT and harness directories
can still be overlaid directly. `floor_z` is where this directory's own clouds have their
floor: 0 for a harness producer (the frame is fitted to that floor), -0.026 m for the
ScanNet GT on scene0000_00 (the GT mesh sits that far from the floor AMB3R reconstructed).

Camera
------
One colour camera for the whole scene: `manifest["camera"]` carries its
intrinsics and image size, and each key frame carries `camera_to_world`, a 4x4
camera-to-world matrix in the scan frame under the OpenCV convention (+X right,
+Y down, +Z forward). With those two a consumer can project any scan-frame point
into a key frame:

    p_cam = inverse(camera_to_world) @ [x, y, z, 1]
    u = fx * p_cam.x / p_cam.z + cx,  v = fy * p_cam.y / p_cam.z + cy

`camera_to_world` is required: a harness producer has an estimated pose for
every frame it lifts, so the field does not break the both-producers rule.
"""
from __future__ import annotations

# v2 (2026-11-10): manifest gains `camera` (intrinsics + extrinsics convention),
# `floor_z` and `axis_alignment`, and every key frame must carry
# `camera_to_world`. A v1 directory fails these checks, so the version moved;
# every v1 field is still written unchanged, so a v1 reader keeps working.
# v3 (2026-11-18): coordinate_frame scannet_scan -> amb3r_floor. `axis_alignment` is gone,
# `frame` is required, `floor_z` is within 10 cm of 0. Every coordinate moved, so a v2 reader that
# assumed the scan frame would place things wrongly; hence the version.
SCHEMA_VERSION = "handoff/v3"
FRAME = "amb3r_floor"
# What a manifest's `frame` block must carry (src.geometry.floor_frame.Frame.block()).
COORD_FRAME_REQUIRED = ("definition", "file", "native_to_frame", "scale", "first_frame")
SCALE_SOURCES = ("oracle_gt_poses", "native")

# Labels that describe the building rather than a thing in it. The object
# generator should skip these by default: nine wall segments are not nine assets.
STRUCTURAL_LABELS = frozenset({
    "wall", "floor", "ceiling", "window", "doorframe",
})

# Labels that name a category but do not say which thing: ScanNet uses `object`
# for the five instances its annotators could not name. A generator handed one of
# these has nothing to build from and should ask for a human label first.
GENERIC_LABELS = frozenset({"object", "objects", "stuff", "unknown"})


def label_quality(label: str) -> str:
    """How much a label tells the object generator. Three cases, not two.

    A single `structural` boolean would have to file `object` as non-structural,
    which is true and useless: those five instances are unbuildable for a different
    reason, and merging the two cases loses the distinction the consumer needs.
    """
    text = (label or "").strip().lower()
    if text in STRUCTURAL_LABELS:
        return "structural"
    if text in GENERIC_LABELS:
        return "generic"
    return "specific"


# Fields every object record must carry. Optional fields are listed separately so
# `check.py` can tell "missing" from "this producer cannot supply it".
OBJECT_REQUIRED = (
    "object_id", "label", "label_vocab", "label_quality", "structural",
    "cloud", "point_count", "centroid", "bounds", "key_frames",
)

# Fields the GT producer writes and a harness producer may leave null.
OBJECT_OPTIONAL = ("description", "mesh", "frames_visible", "semantic_id")

FRAME_REQUIRED = (
    "seq", "source_frame_id", "rgb", "mask", "boxed",
    "bbox_xyxy", "bbox_convention", "visible_pixels", "pixel_ratio",
    "touches_border", "role", "camera_to_world",
)

# Scene-level fields every manifest must carry, checked by `check.py`.
MANIFEST_REQUIRED = (
    "schema_version", "scene_id", "producer", "coordinate_frame", "units", "up_axis",
    "camera", "floor_z", "frame",
)
CAMERA_REQUIRED = ("model", "width", "height", "fx", "fy", "cx", "cy", "K",
                   "extrinsics")
# What `camera_to_world` means, written verbatim into every manifest so the
# convention travels with the data instead of living in a separate document.
EXTRINSICS_CONVENTION = {
    "field": "key_frames[].camera_to_world",
    "direction": "camera_to_world",
    "axes": "opencv",
    "axes_detail": "+X right, +Y down, +Z forward (optical axis); not OpenGL",
    "frame": FRAME,
    "units": "meters",
    "layout": "4x4 row-major; rotation in [:3,:3], camera centre in [:3,3]",
}

# `xmax`/`ymax` exclusive, matching `harness/observations.py` so a GT box and a
# harness observation box can be compared without a conversion step.
BBOX_CONVENTION = "original RGB pixels; origin top-left; xmax/ymax exclusive"

KEYFRAME_ROLES = ("primary", "secondary", "third")


def empty_object(object_id: str) -> dict:
    """A record with every required key present, so writers cannot forget one."""
    return {
        "object_id": object_id,
        "label": "",
        "label_vocab": "scannet",
        "label_quality": "specific",
        "description": None,
        "structural": False,
        "cloud": "",
        "mesh": None,
        "point_count": 0,
        "centroid": [0.0, 0.0, 0.0],
        "bounds": {"minimum": [0.0, 0.0, 0.0], "maximum": [0.0, 0.0, 0.0]},
        "frames_visible": 0,
        "semantic_id": None,
        "key_frames": [],
    }


def empty_frame() -> dict:
    return {
        "seq": 0,
        "source_frame_id": 0,
        "rgb": "",
        "mask": "",
        "boxed": "",
        "bbox_xyxy": [0, 0, 0, 0],
        "bbox_convention": BBOX_CONVENTION,
        "visible_pixels": 0,
        "pixel_ratio": 0.0,
        "bbox_fill": 0.0,
        "clarity": 0.0,
        # How much of this object's best frame this one shows. Below 1.0 the
        # object is partly occluded; the selection floor is 0.20.
        "visibility": 0.0,
        "touches_border": False,
        "camera_center": None,
        "camera_to_world": None,
        "view_angle_deg": None,
        "role": "primary",
    }


def camera(width: int, height: int, fx: float, fy: float, cx: float, cy: float,
           source: str) -> dict:
    """The manifest's `camera` block: pinhole intrinsics plus the extrinsic convention."""
    return {
        "model": "pinhole",
        "width": int(width), "height": int(height),
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "K": [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        "pixel_convention": "u right, v down, origin at the top-left pixel; "
                            "same image as key_frames[].rgb",
        "distortion": None,
        "intrinsics_source": source,
        "extrinsics": dict(EXTRINSICS_CONVENTION),
    }


def manifest(scene_id: str, producer: dict, **extra) -> dict:
    record = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "producer": producer,
        "coordinate_frame": FRAME,
        "units": "meters",
        "up_axis": "Z",
        "bbox_convention": BBOX_CONVENTION,
        "structural_labels": sorted(STRUCTURAL_LABELS),
        "generic_labels": sorted(GENERIC_LABELS),
    }
    record.update(extra)
    return record


def keyframe_policy(pool: str, occlusion_floor: float, occlusion_floor_structural: float,
                    edge_penalty: float, separation_deg, max_views: int,
                    min_frame_gap: int) -> dict:
    """The key-frame selection rule, built from the producer's own constants.

    It used to be a literal here, and it drifted: it listed three separation
    steps while `from_gt.py` tried four, and it omitted the frame-gap rule and the
    structural floor. Taking the values as arguments keeps the description equal
    to what actually ran.
    """
    return {
        "pool": pool,
        "order": [f"drop frames showing <{occlusion_floor:.0%} of the object's best view "
                  f"(<{occlusion_floor_structural:.0%} for structural labels)",
                  f"rank by score, with edge contact costing {1 - edge_penalty:.0%}",
                  "then widen view separation until three distinct views are found"],
        "score": "pixel_ratio * (0.5 + 0.5 * min(clarity / median_clarity, 2))",
        "occlusion_floor": occlusion_floor,
        "occlusion_floor_structural": occlusion_floor_structural,
        "edge_penalty": edge_penalty,
        "separation_deg": list(separation_deg),
        "min_frame_gap": min_frame_gap,
        "max_views": max_views,
        "border_rule": f"bbox touching the image edge costs {1 - edge_penalty:.0%} of the score; "
                       "it never outranks how much of the object is visible",
    }
