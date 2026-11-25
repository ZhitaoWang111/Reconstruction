import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objgen.appearance.multiview import discover

class OrderedViews(unittest.TestCase):
    def test_order_and_optional_poles(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for n in ('down.png','180.jpg','0.jpg','270.jpg','up.jpg','90.jpg','notes.txt'): (p/n).touch()
            self.assertEqual([a for a,_ in discover(p)],['0','90','180','270','up','down'])
    def test_missing_front_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'90.jpg').touch()
            with self.assertRaises(ValueError):discover(d)
    def test_duplicate_angle_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ('0.jpg','0.png'):(Path(d)/n).touch()
            with self.assertRaises(ValueError):discover(d)

if __name__=='__main__':unittest.main()
