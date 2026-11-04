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
Points are in the ScanNet *scan* frame: metres, gravity-aligned, +Z up, no
`axisAlignment` applied. That is the frame the AMB3R source was Sim(3)-aligned
into (median surface distance 0.12 m against 2.97 m for the axis-aligned mesh),
so GT clouds and harness clouds can be overlaid directly. The axis alignment
matrix is recorded in the manifest for anyone who needs the NYU frame instead.
"""
from __future__ import annotations

SCHEMA_VERSION = "handoff/v1"

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
    "touches_border", "role",
)

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
        "view_angle_deg": None,
        "role": "primary",
    }


def manifest(scene_id: str, producer: dict, **extra) -> dict:
    record = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "producer": producer,
        "coordinate_frame": "scannet_scan",
        "units": "meters",
        "up_axis": "Z",
        "bbox_convention": BBOX_CONVENTION,
        "structural_labels": sorted(STRUCTURAL_LABELS),
        "generic_labels": sorted(GENERIC_LABELS),
        # Recorded so a consumer can reproduce the choice instead of trusting it.
        "keyframe_policy": {
            "pool": "500-frame harness sample",
            "order": ["drop frames showing <20% of the object's best view",
                      "rank by score, with edge contact costing 15%",
                      "then widen view separation until three distinct views are found"],
            "score": "pixel_ratio * (0.5 + 0.5 * min(clarity / median_clarity, 2))",
            "occlusion_floor": 0.20,
            "edge_penalty": 0.85,
            "separation_deg": [20.0, 14.0, 10.0],
            "max_views": 3,
            "border_rule": "bbox touching the image edge costs 15% of the score; "
                           "it never outranks how much of the object is visible",
        },
    }
    record.update(extra)
    return record
