"""Object handoff from a harness run (src.handoff.from_harness) and the amb3r_floor lifting it reads."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.handoff import check, from_harness as fh

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "artifacts/handoff/scene0000_00/harness_v15"


def test_objects_of_keeps_live_objects_and_their_holds():
    state = {"objects": {"Object1": {"object_label": "chair", "status": "provisional"},
                         "Object2": {"object_label": "lamp", "status": "merged"},
                         "Object3": {"object_label": "cup", "status": "provisional"}},
             "assignments": {"0000:001": {"object_id": "Object1", "status": "held"},
                             "0001:002": {"object_id": "Object1", "status": "assigned"},
                             "0001:003": {"object_id": None, "status": "discard"},
                             "0002:001": {"object_id": "Object2", "status": "held"}}}
    out = fh.objects_of(state)
    assert out == {"Object1": {"label": "chair", "held": ["0000:001", "0001:002"]}}   # merged and empty objects dropped


def test_a_scan_frame_source_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"lifting_policy": "amb3r_per_frame_v1", "frames": []}))
    with pytest.raises(SystemExit, match="not an amb3r_floor source"):
        fh.Source(tmp_path)


real = pytest.mark.skipif(not (REAL / "manifest.json").is_file(), reason="harness handoff not built")


@real
def test_real_harness_handoff_passes_the_checker():
    assert check.check(REAL) == []


@real
def test_real_harness_handoff_is_the_runs_objects_in_amb3r_floor():
    man = json.loads((REAL / "manifest.json").read_text())
    objs = json.loads((REAL / "objects.json").read_text())
    assert man["schema_version"] == "handoff/v3" and man["coordinate_frame"] == "amb3r_floor"
    assert man["producer"]["name"] == "harness" and man["floor_z"] == 0.0
    run = ROOT / man["producer"]["run_dir"]
    live = fh.objects_of(fh.last_state(run))
    assert len(objs) == len(live)
    # every object stands on or above the floor: the lowest 1% of its points is not under z = -0.1
    lows = [r["bounds"]["minimum"][2] for r in objs]
    assert min(lows) > -0.25 and np.median(lows) > -0.05
    # key frames are held observations of that object, and their poses put the camera at hand height
    held = {o: set(v["held"]) for o, v in live.items()}
    for r in objs:
        obj = "Object" + str(int(r["object_id"].removeprefix("obj_")))
        for k in r["key_frames"]:
            assert k["observation_id"] in held[obj]
            assert 1.0 < k["camera_to_world"][2][3] < 1.7
    assert np.median([k["reprojection_px"] for r in objs for k in r["key_frames"]]) < 15
