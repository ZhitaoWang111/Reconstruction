"""The wall handoff contract: what the wall generator reads to build the room's walls.

Walls, and the openings in them. Doors and windows are not separate items in it (their
leaves and panes are elsewhere in the pipeline), but a hole a wall has to leave for them
is: `openings` lists what was *measured* to be a hole, not what is expected to be one.

An opening is found by the ray test - rays see through the wall from floor to top, with
solid wall above (`closed_above`). That finds doorways and passages that stand open; a
closed door leaf or a window pane stops the rays, so those openings are *not* in this
version and a consumer must not read `openings: []` as "a solid wall here".

A sibling of the object handoff (`schema.py`, handoff/v2), and deliberately shaped
like it so one reader can open both:

* the same coordinate frame - ScanNet *scan* frame, metres, +Z up, no axisAlignment;
* the same `camera` block (pinhole intrinsics + the OpenCV camera-to-world convention);
* the same key-frame record - up to three frames, roles primary/secondary/third,
  each with `rgb`, `mask`, `boxed`, `bbox_xyxy`, `visible_pixels`, `camera_to_world`.

Geometry
--------
A wall is its *room-side face* drawn on the floor: `start` -> `end`, plus
`normal_to_back` (unit, horizontal, pointing away from the room). Thickness is laid
behind that face, so changing a thickness never moves the room's inner surface.
`start`, `end`, `normal_to_back`, `thickness_m` and `height_m` define the wall
completely; `footprint` is the same thing pre-computed (corners mitred where the
`joins` say so) for consumers that just want a prism. A consumer that changes a
thickness must recompute the footprint from the line, not edit the footprint.

All coordinates are scan frame. The floor frame (z = 0 on the floor, walls vertical)
is often easier to build in; `floor_frame_to_scan` maps it to the scan frame:
`p_scan = floor_frame_to_scan @ [x, y, z, 1]`. Every 2D point in this file is the
(x, y) of a floor-frame point at z = 0, and every 3D point is scan frame - the field
name says which (`*_floor` or `*_scan`).

Observed or not
---------------
Every value that was not measured says so: `kind: assumed` (storey height,
thickness) or a wall segment with `kind: inferred` (extended past what any frame
saw). Evidence that hints at a value but was not used to set it lives under
`evidence` and never overrides the value.
"""
from __future__ import annotations

from .schema import KEYFRAME_ROLES

SCHEMA_VERSION = "walls/v1"

ROLES = ("exterior", "interior")
SEGMENT_KINDS = ("observed", "inferred")
JOIN_TYPES = ("mitre", "butt", "none")
VALUE_KINDS = ("measured", "assumed")

NOT_INCLUDED = ("doors", "windows", "door_frames", "window_frames", "ceiling", "floor_mesh",
                "mirrors", "furniture")

OPENING_SOURCES = ("ray_test",)

TOP_REQUIRED = (
    "schema_version", "scene_id", "producer", "coordinate_frame", "units", "up_axis",
    "floor_frame_to_scan", "camera", "storey", "thickness_defaults", "outline",
    "walls", "not_included", "known_issues",
)

WALL_REQUIRED = (
    "id", "role", "start_floor", "end_floor", "length_m", "normal_to_back",
    "thickness_m", "thickness_kind", "height_m", "footprint_floor", "segments",
    "joins", "openings", "key_frames", "evidence",
)

SEGMENT_REQUIRED = ("from_m", "to_m", "kind", "source_planes")

# An opening, as measured: along the wall from `start`, between floor and its head, and
# the same hole in the scan frame. `kind` is opening, not door or window, because the ray
# test shows a hole and nothing about what fills it.
OPENING_REQUIRED = ("from_m", "to_m", "z_floor", "width_m", "height_m", "kind", "source",
                    "rays_through", "closed_above", "corners_floor", "corners_scan")

KEYFRAME_REQUIRED = (
    "role", "seq", "source_frame_id", "rgb", "mask", "boxed", "bbox_xyxy",
    "bbox_convention", "visible_pixels", "covers_m", "view_angle_deg", "distance_m",
    "camera_to_world",
)

KEYFRAME_OPTIONAL = ("rectified",)

RECTIFIED_REQUIRED = ("image", "px_per_m", "along_m", "height_m")

# Thresholds a key frame must clear; stated here because both the producer and the
# checker use them, and the README quotes them.
MAX_KEYFRAMES = len(KEYFRAME_ROLES)
