"""Thresholds and levels for M4. All in metres or degrees; one file per run is written
next to the results so runs stay comparable.

Confirmed by the user on 2026-11-20 (values cross-checked against the two reference
implementations: LiteReality's geometry.py constants and scenesmith's
physics_validation/physical_feasibility defaults).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # "not worse than before": a violation may grow by at most this much and stay
    tolerance_m: float = 0.01

    # walls (visual mesh against the analytic wall body)
    wall_report_m: float = 0.01
    wall_stop_m: float = 0.02

    # leaving the room (footprint against the floor polygons)
    room_report_m: float = 0.02
    room_stop_m: float = 0.05

    # contact with the declared support: gap under it, or sinking into it
    support_float_report_m: float = 0.01
    support_float_stop_m: float = 0.05
    support_sink_report_m: float = 0.01
    support_sink_stop_m: float = 0.03
    support_contact_ratio_min: float = 0.20      # scenesmith's min_area_ratio

    # declared against a wall: gap to the wall face
    wall_gap_report_m: float = 0.01
    wall_gap_stop_m: float = 0.05

    # physics (Isaac settle)
    phys_report_move_m: float = 0.05
    phys_fall_tilt_deg: float = 45.0             # scenesmith's fallen_tilt_threshold
    phys_drop_m: float = 0.30                    # scenesmith's fallen_manipuland_z_displacement
    phys_near_floor_m: float = 0.02

    # asset itself: only reported
    scale_ok: tuple = (0.5, 2.0)

    # scan overlap: only reported
    cloud_tol_m: float = 0.05

    # objects that sit in a wall opening by nature: wall and room checks do not apply (D07-3).
    # Matched on the label, so it carries over to other scenes.
    wall_exempt_labels: tuple = ('door', 'window')

    # consecutive automatic undos before the object is marked failed
    max_undo_streak: int = 3

    # penetration thresholds are not used while penetration is a yes/no answer;
    # kept here for the day a depth becomes available
    penetration_report_m: float | None = None
    penetration_stop_m: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['scale_ok'] = list(self.scale_ok)
        d['wall_exempt_labels'] = list(self.wall_exempt_labels)
        return d

    def write(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))
