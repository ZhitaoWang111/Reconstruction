"""Packaging checks run in the USD/SAM3D environment."""
import tempfile
from pathlib import Path
import unittest
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objgen.appearance.package_asset import package, physical_snapshot


class PackageTests(unittest.TestCase):
    def fixture(self,folder):
        stage=Usd.Stage.CreateNew(str(folder/'visual.usda'))
        root=UsdGeom.Xform.Define(stage,'/Object').GetPrim();stage.SetDefaultPrim(root)
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1)
        mesh=UsdGeom.Mesh.Define(stage,'/Object/Mesh');mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
        mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2]);mesh.CreateDisplayColorAttr([Gf.Vec3f(.5)])
        material=UsdShade.Material.Define(stage,'/Object/Looks/Visual')
        tex=UsdShade.Shader.Define(stage,'/Object/Looks/Visual/Texture');tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./source.png'))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        (folder/'source.png').write_bytes(b'texture fixture')
        stage.GetRootLayer().Save()
        UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
        mass=UsdPhysics.MassAPI.Apply(root);mass.CreateMassAttr(.5);mass.CreateCenterOfMassAttr(Gf.Vec3f(0,0,.1))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        pm=UsdShade.Material.Define(stage,'/Object/Looks/Physics')
        UsdPhysics.MaterialAPI.Apply(pm.GetPrim()).CreateStaticFrictionAttr(.7)
        UsdShade.MaterialBindingAPI(mesh).Bind(pm,materialPurpose='physics')
        UsdPhysics.SetStageKilogramsPerUnit(stage,1)
        stage.GetRootLayer().Export(str(folder/'physics.usda'))

    def test_relocated_closure_and_exact_physics(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p)
            result=package(p/'visual.usda',p/'physics.usda',p/'packed')
            self.assertTrue(result['passed'])
            shutil.move(p/'packed',p/'moved');(p/'source.png').unlink();(p/'visual.usda').unlink()
            stage=Usd.Stage.Open(str(p/'moved/object.usda'))
            tex=UsdShade.Shader(stage.GetPrimAtPath('/Object/Looks/Visual/Texture')).GetInput('file').Get()
            self.assertTrue(Path(tex.resolvedPath).is_file())
            self.assertEqual(physical_snapshot(stage),physical_snapshot(Usd.Stage.Open(str(p/'physics.usda'))))

    def test_missing_dependency_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p);(p/'source.png').unlink()
            with self.assertRaises(AssertionError):package(p/'visual.usda',p/'physics.usda',p/'packed')

    def test_geometry_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p)
            stage=Usd.Stage.Open(str(p/'visual.usda'));UsdGeom.Mesh(stage.GetPrimAtPath('/Object/Mesh')).GetPointsAttr().Set([(0,0,0),(2,0,0),(0,1,0)]);stage.GetRootLayer().Save()
            with self.assertRaises(AssertionError):package(p/'visual.usda',p/'physics.usda',p/'packed')


if __name__=='__main__':unittest.main()
