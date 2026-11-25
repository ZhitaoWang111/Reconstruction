"""Stage 1 without Blender: the segment split and the prism each segment builds."""
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structuregen import config
from structuregen.common import handoff as handoff_io
from structuregen.whitebox import segments


def wall(wid, start, end, thickness=0.3, height=2.8, joins=None, openings=(), normal=None):
    s, e = start, end
    length = math.dist(s, e)
    u = ((e[0] - s[0]) / length, (e[1] - s[1]) / length)
    n = normal or (-u[1], u[0])           # left-hand side of start -> end is the room
    fp = [list(s), list(e),
          [e[0] + n[0] * thickness, e[1] + n[1] * thickness],
          [s[0] + n[0] * thickness, s[1] + n[1] * thickness]]
    return {"id": wid, "role": "interior", "start_floor": list(s), "end_floor": list(e),
            "normal_to_back": [n[0], n[1], 0.0], "thickness_m": thickness, "height_m": height,
            "length_m": length, "footprint_floor": fp, "joins": joins or {},
            "openings": list(openings)}


def doc(*walls, scene="test"):
    return {"schema_version": "walls/v2", "scene_id": scene, "coordinate_frame": "amb3r_floor",
            "walls": list(walls)}


class PlanTests(unittest.TestCase):
    def test_butt_cut_splits_the_host_at_the_centre_line(self):
        host = wall("ext", (0.0, 0.0), (6.0, 0.0))
        # a wall butting onto the host's room side at along 2.0, 0.3 m thick
        other = wall("div", (2.0, 0.0), (2.0, 1.0), thickness=0.15,
                     joins={"start": {"type": "butt", "wall": "ext"}})
        planned = segments.plan(doc(host, other))
        ids = [s["id"] for s in planned["segments"]]
        self.assertEqual(ids, ["ext_1", "ext_2", "div"])
        first = planned["segments"][0]
        self.assertAlmostEqual(first["a1"], 1.925)     # centre line of the 0.15 m wall
        self.assertAlmostEqual(planned["segments"][1]["a0"], 1.925)

    def test_mitre_is_not_cut(self):
        a = wall("a", (0.0, 0.0), (6.0, 0.0), joins={"end": {"type": "mitre", "wall": "b"}})
        b = wall("b", (6.0, 0.0), (6.0, 4.0), joins={"start": {"type": "mitre", "wall": "a"}})
        planned = segments.plan(doc(a, b))
        self.assertEqual(sorted(s["id"] for s in planned["segments"]), ["a", "b"])

    def test_segment_volume_is_footprint_times_height_minus_openings(self):
        w = wall("w", (0.0, 0.0), (6.0, 0.0), thickness=0.3, height=2.8,
                 openings=[{"from_m": 1.0, "to_m": 2.0, "z_floor": [0.0, 1.8]}])
        seg = segments.plan(doc(w))["segments"][0]
        self.assertAlmostEqual(seg["expected_volume_m3"], 6.0 * 0.3 * 2.8 - 1.0 * 1.8 * 0.3)

    def test_opening_straddling_a_cut_is_reported(self):
        host = wall("ext", (0.0, 0.0), (6.0, 0.0),
                    openings=[{"from_m": 1.9, "to_m": 2.3, "z_floor": [0.0, 1.8]}])
        other = wall("div", (2.075, 0.0), (2.075, 1.0), thickness=0.15,
                     joins={"start": {"type": "butt", "wall": "ext"}})
        planned = segments.plan(doc(host, other))
        # the opening lands in no segment; it must be reported, not silently left solid
        self.assertEqual(len(planned["dropped_openings"]), 1)
        self.assertTrue(any("straddles" in p for p in segments.check(planned)))

    def test_non_positive_volume_is_reported(self):
        w = wall("w", (0.0, 0.0), (1.0, 0.0), thickness=0.1, height=0.5,
                 openings=[{"from_m": 0.05, "to_m": 0.95, "z_floor": [0.0, 0.6]}])
        planned = segments.plan(doc(w))
        self.assertTrue(segments.check(planned))

    def test_height_override_applies_to_every_segment(self):
        w = wall("w", (0.0, 0.0), (2.0, 0.0), height=2.8)
        seg = segments.plan(doc(w), height_m=2.37)["segments"][0]
        self.assertAlmostEqual(seg["height_m"], 2.37)
        self.assertAlmostEqual(seg["expected_volume_m3"], 2.0 * 0.3 * 2.37)


class HandoffTests(unittest.TestCase):
    def test_reads_the_real_handoff_when_present(self):
        if not (config.DEFAULT_HANDOFF / "walls.json").is_file():
            self.skipTest("handoff not present")
        doc_ = handoff_io.load(config.DEFAULT_HANDOFF)
        planned = segments.plan(doc_)
        self.assertEqual(len(planned["segments"]), 9)
        ids = [s["id"] for s in planned["segments"]]
        self.assertEqual(ids, ["ext_W", "ext_S", "ext_E_1", "ext_E_2", "ext_N_1", "ext_N_2",
                               "int_1", "int_2", "int_3"])
        self.assertEqual(segments.check(planned), [])
        int3 = next(s for s in planned["segments"] if s["id"] == "int_3")
        self.assertEqual(len(int3["openings"]), 1)

    def test_rejects_another_schema(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "walls.json").write_text(json.dumps(
                {"schema_version": "walls/v1", "coordinate_frame": "amb3r_floor", "walls": []}))
            with self.assertRaises(ValueError):
                handoff_io.load(Path(tmp))


if __name__ == "__main__":
    unittest.main()
