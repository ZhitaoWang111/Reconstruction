"""Author a relative base-colour texture without changing canonical geometry."""
import json
from pathlib import Path
import sys
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt
folder=Path(sys.argv[1]); manifest=json.loads((folder/'manifest.json').read_text())
stage=Usd.Stage.Open(manifest['inputs']['usd']['path'])
stage.GetRootLayer().Export(str(folder/'object.usda'))
stage=Usd.Stage.Open(str(folder/'object.usda'))
root=stage.GetDefaultPrim(); mesh=UsdGeom.Mesh(stage.GetPrimAtPath(root.GetPath().AppendChild('Mesh')))
uv=np.load(folder/'uv.npy').reshape(-1,2)
assert len(uv)==len(mesh.GetFaceVertexIndicesAttr().Get()) and np.isfinite(uv).all()
pv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying)
pv.Set(Vt.Vec2fArray.FromNumpy(uv))
original_material=UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
material=UsdShade.Material.Define(stage,root.GetPath().AppendPath('Looks/PhotoGrounded'))
shader=UsdShade.Shader.Define(stage,material.GetPath().AppendChild('Surface')); shader.CreateIdAttr('UsdPreviewSurface')
shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.45)
reader=UsdShade.Shader.Define(stage,material.GetPath().AppendChild('UV')); reader.CreateIdAttr('UsdPrimvarReader_float2'); reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
texture=UsdShade.Shader.Define(stage,material.GetPath().AppendChild('BaseColor')); texture.CreateIdAttr('UsdUVTexture')
texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./textures/basecolor.png'))
texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput('result',Sdf.ValueTypeNames.Float2))
shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3))
material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput('surface',Sdf.ValueTypeNames.Token))
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
observed=np.load(folder/'observed_faces.npy')
subset=UsdShade.MaterialBindingAPI(mesh).CreateMaterialBindSubset('UnobservedOriginal',Vt.IntArray.FromNumpy(np.flatnonzero(~observed).astype(np.int32)))
UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(original_material)
stage.GetRootLayer().Save()
# Reopen, verify arrays and resolved texture rather than relying on write success.
stage=Usd.Stage.Open(str(folder/'object.usda')); mesh=UsdGeom.Mesh(stage.GetPrimAtPath(mesh.GetPath()))
arrays=np.load(folder/'canonical_mesh.npz')
for key,attr in [('points',mesh.GetPointsAttr()),('counts',mesh.GetFaceVertexCountsAttr()),('indices',mesh.GetFaceVertexIndicesAttr()),('colors',mesh.GetDisplayColorPrimvar())]:
    assert np.array_equal(np.asarray(attr.Get()),arrays[key]),key
assert UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0].GetPath()==material.GetPath()
asset=UsdShade.Shader(stage.GetPrimAtPath(texture.GetPath())).GetInput('file').Get()
assert asset.resolvedPath and Path(asset.resolvedPath).is_file()
assert stage.GetDefaultPrim().GetPath()==root.GetPath()
(folder/'usd_validation.json').write_text(json.dumps({'passed':True,'geometry_arrays_exact':True,'face_varying_uvs':len(uv),'relative_texture':asset.path,'up_axis':str(UsdGeom.GetStageUpAxis(stage)),'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'isaac_verified':False},indent=2))
