"""The appearance session against a real headless Blender: gates, rollback, checks, compare.

Skipped when Blender or a built run (structuregen.whitebox.build + structuregen.evidence.package) is missing.
"""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structuregen import config  # noqa: E402

RUN = (config.OUTPUT_ROOT / config.SCENE / "structuregen/20261119_134429")   # the walls_harness run: fixtures name its walls and frames
READY = config.BLENDER.is_file() and (RUN / "evidence/task_card.json").is_file()

PAINT = "import bpy\nbpy.ops.sg.paint(seg='ext_E_1', face='front', rgb=(200,192,178), item='t1')"
RAIL = "import bpy\nbpy.ops.sg.strip(seg='ext_E_1', z0=0.96, z1=1.14, depth={d}, rgb=(176,128,74), kind='rail', item='t2')"
DECLARE = {"surface": {"rgb": [200, 192, 178], "basis": [894]},
           "trims": [{"kind": "rail", "z": [0.96, 1.14], "basis": [894]}]}


@unittest.skipUnless(READY, "needs Blender and a built structuregen run")
class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structuregen.agent.session import Session
        cls.tmp = tempfile.TemporaryDirectory()
        cls.s = Session(RUN.resolve(), Path(cls.tmp.name) / "s")

    @classmethod
    def tearDownClass(cls):
        cls.s.close()
        cls.tmp.cleanup()

    def test_1_edit_refused_before_declare(self):
        r = self.s.execute_blender_code(PAINT, "paint")
        self.assertTrue(r.get("refused"))

    def test_2_declare_checks_frames_and_kinds(self):
        r = self.s.declare(3, {"surface": {"basis": [1229]}, "trims": []})
        self.assertTrue(r.get("refused"))
        r = self.s.declare(3, {"surface": {"basis": [894]}, "trims": [{"kind": "window", "basis": [894]}]})
        self.assertTrue(r.get("refused"))
        r = self.s.declare(3, DECLARE)
        self.assertFalse(r.get("refused"))
        self.assertEqual(sorted(self.s.walls["ext_E_1"]["items"]), ["t1", "t2"])

    def test_3_safe_mode(self):
        r = self.s.execute_blender_code("import os\nos.listdir('.')", "x")
        self.assertTrue(r.get("refused"))

    def test_4_rollbacks(self):
        r = self.s.execute_blender_code(PAINT + "\nbpy.ops.sg.paint(seg='ext_E_1', face='side', rgb=(1,1,1), item='t1')", "x")
        self.assertTrue(r["rolled_back"])
        self.assertIn("line 3", r["text"])
        self.assertIn("front", self.s._progress("ext_E_1")["unpainted"])         # the first paint was undone too
        r = self.s.execute_blender_code("import bpy\nbpy.ops.sg.paint(seg='ext_W', face='front', rgb=(1,1,1), item='t1')", "x")
        self.assertTrue(r["rolled_back"])
        r = self.s.execute_blender_code("import bpy\nbpy.data.objects['ext_E_1'].data.vertices[0].co.z += 0.3", "x")
        self.assertTrue(r["rolled_back"])
        r = self.s.execute_blender_code("import bpy\nbpy.data.objects.new('junk', None)\n"
                                        "bpy.context.scene.collection.objects.link(bpy.data.objects['junk'])", "x")
        self.assertTrue(r["rolled_back"])

    def test_5_trim_problem_blocks_finish_then_passes(self):
        r = self.s.execute_blender_code(PAINT + "\n" + RAIL.format(d=0.08).split("\n", 1)[1], "paint, thick rail")
        self.assertFalse(r["rolled_back"])
        self.assertTrue(any("sticks out" in t for t in r["trims"]))
        self.assertTrue(Path(r["image"]).is_file())
        r = self.s.finish_segment(3, "x")
        self.assertTrue(r.get("refused"))
        r = self.s.execute_blender_code(RAIL.format(d=0.02), "rail 2 cm")
        self.assertEqual(r["trims"], [])
        r = self.s.finish_segment(3, "off-white wall, wood rail")
        self.assertFalse(r.get("refused"), r["text"])
        self.assertEqual(self.s.current, "ext_W")

    def test_6_now_on_the_next_wall(self):
        r = self.s.execute_blender_code(PAINT, "paint the finished wall")
        self.assertTrue(r.get("refused") or r.get("rolled_back"))
        r = self.s.compare(894)                                                  # not a key frame of ext_W
        self.assertTrue(r.get("refused"))


if __name__ == "__main__":
    unittest.main()
