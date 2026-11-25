"""Camera/UV coordinate checks independent of Blender and model APIs."""
import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objgen.appearance.camera_fit import basis, project, apply_overrides


class ProjectionTests(unittest.TestCase):
    def test_basis_is_right_handed_orthonormal(self):
        for e,r,y in [(0,0,0),(19,2,70),(-12,-7,180)]:
            b=basis(e,r,y)
            np.testing.assert_allclose(b@b.T,np.eye(3),atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(b),1)

    def test_override_rejects_invalid_basis_and_unknown_fields(self):
        camera={'center':[0,0,0],'basis':basis(0).tolist(),'height_m':.14,'aspect':.5,'offset':[.5,.5],'elevation_degrees':0,'roll_degrees':0,'yaw_degrees':0}
        for override in ({'basis':[[1,0,0]]*3},{'height_m':float('nan')},{'python':'evil'},{'yaw_degrees':200},{'offset':[2,.5]}):
            with self.assertRaises(ValueError):apply_overrides(camera,override)
        fitted=apply_overrides(camera,{'yaw_degrees':30,'scale_factor':1.1})
        self.assertAlmostEqual(fitted['height_m'],.154)
        np.testing.assert_allclose(fitted['basis'],basis(0,0,30))

    def test_photo_origin_aspect_and_axes(self):
        camera={'center':[0,0,.07],'basis':basis(0).tolist(),'height_m':.14,'aspect':.5,'offset':[.5,.5]}
        uv=project(np.array([[0,0,.07],[.035,0,.07],[0,0,.14],[0,-.1,.07]]),camera)
        np.testing.assert_allclose(uv,[[.5,.5],[1,.5],[.5,1],[.5,.5]],atol=1e-14)


if __name__=='__main__': unittest.main()

class KnownViewAngles(unittest.TestCase):
    def test_named_axes_and_poles(self):
        from objgen.appearance.camera_fit import basis
        import numpy as np
        for yaw,expected in [(0,[0,-1,0]),(90,[1,0,0]),(180,[0,1,0]),(270,[-1,0,0])]:
            np.testing.assert_allclose(basis(0,yaw=yaw)[2],expected,atol=1e-10)
        for elevation,expected in [(90,[0,0,1]),(-90,[0,0,-1])]:
            B=basis(elevation)
            np.testing.assert_allclose(B[2],expected,atol=1e-10)
            np.testing.assert_allclose(B@B.T,np.eye(3),atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(B),1)

class PoleFramingTests(unittest.TestCase):
    def test_tall_cylinder_top_uses_diameter(self):
        from objgen.appearance.camera_fit import fit
        import cv2
        theta=np.linspace(0,2*np.pi,320)
        radius=np.linspace(0,.02,80)
        x=(radius[:,None]*np.cos(theta)).ravel()
        y=(radius[:,None]*np.sin(theta)).ravel()
        points=np.vstack([np.column_stack([x,y,np.full_like(x,z)]) for z in (0,.14)])
        mask=np.zeros((200,200),np.uint8);cv2.circle(mask,(100,100),90,1,-1)
        camera=fit(points,mask,elevation=90)
        self.assertLess(camera['height_m'],.06)
        self.assertGreater(camera['initializer_iou'],.9)
