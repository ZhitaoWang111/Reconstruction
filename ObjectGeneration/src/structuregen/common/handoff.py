"""Read the upstream walls handoff (walls/v2) and check the parts this stage relies on."""
from __future__ import annotations

from pathlib import Path
import json

SUPPORTED = ("walls/v2",)
FRAMES = ("amb3r_floor", "scannet_scan")     # both put the floor at z = 0 (scannet_scan: z minus frame.floor_z)
ZONES = ("front", "back", "top", "bottom", "end_start", "end_end", "reveal")


def load(handoff: Path) -> dict:
    """The handoff's walls.json, after checking the fields stage 1 reads."""
    handoff = Path(handoff)
    path = handoff / "walls.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    doc = json.loads(path.read_text())
    if doc.get("schema_version") not in SUPPORTED:
        raise ValueError(f"{path}: schema {doc.get('schema_version')!r}, expected one of {SUPPORTED}")
    if doc.get("coordinate_frame") not in FRAMES:
        raise ValueError(f"{path}: frame {doc.get('coordinate_frame')!r}; built frames are {FRAMES}")
    if doc["storey"].get("floor_z_floor", 0.0) != 0.0:
        raise ValueError(f"{path}: storey.floor_z_floor is {doc['storey']['floor_z_floor']}, walls are built from z = 0")
    ids = [w["id"] for w in doc["walls"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate wall ids")
    for w in doc["walls"]:
        for name in ("start_floor", "end_floor", "normal_to_back", "footprint_floor",
                     "thickness_m", "height_m", "joins", "openings"):
            if name not in w:
                raise ValueError(f"{path}: wall {w['id']} lacks {name}")
        if len(w["footprint_floor"]) != 4:
            raise ValueError(f"{path}: wall {w['id']} footprint_floor is not a quad")
        for o in w["openings"]:
            for name in ("from_m", "to_m", "z_floor"):
                if name not in o:
                    raise ValueError(f"{path}: wall {w['id']} opening lacks {name}")
    for w in doc["walls"]:              # GT-pose handoffs carry mask_agreement instead of a reprojection error
        for k in w["key_frames"]:
            k.setdefault("reprojection_px", None)
    doc["_path"] = str(path)
    return doc


def opening_segment(wall: dict, opening: dict, spans: list[tuple[float, float]]) -> str | None:
    """The segment id whose span strictly contains the opening, or None when it straddles a cut.

    An opening must lie inside one segment: a hole crossing a cut would need the two bodies
    to agree on its edges, and no handoff opening does.
    """
    for a0, a1, sid in spans:
        if a0 < opening["from_m"] and opening["to_m"] < a1:
            return sid
    return None
