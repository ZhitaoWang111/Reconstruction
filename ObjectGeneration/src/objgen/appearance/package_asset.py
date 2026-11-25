"""Preserve baseline physics exactly and collect the textured USD dependency closure.

Run with the USD Python environment; no Isaac Sim imports required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_snapshot(stage):
    result={}
    for prim in stage.Traverse():
        schemas=[str(s) for s in prim.GetAppliedSchemas() if 'Physics' in s or 'Physx' in s]
        props={}
        for attr in prim.GetAttributes():
            if str(attr.GetName()).startswith(('physics:','physx')):
                props[str(attr.GetName())]=str(attr.Get())
        rels={str(r.GetName()):[str(t) for t in r.GetTargets()] for r in prim.GetRelationships()
              if str(r.GetName()).startswith(('physics:','material:binding:physics'))}
        if schemas or props or rels:
            result[str(prim.GetPath())]={'schemas':sorted(schemas),'attributes':props,'relationships':rels}
    result['kilogramsPerUnit']=stage.GetMetadata('kilogramsPerUnit')
    return result


def geometry_equal(left,right):
    assert left.GetDefaultPrim().GetPath()==right.GetDefaultPrim().GetPath()
    assert UsdGeom.GetStageUpAxis(left)==UsdGeom.GetStageUpAxis(right)=='Z'
    assert UsdGeom.GetStageMetersPerUnit(left)==UsdGeom.GetStageMetersPerUnit(right)==1
    for prim in left.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            l=UsdGeom.Mesh(prim); r=UsdGeom.Mesh(right.GetPrimAtPath(prim.GetPath())); assert r
            for name in ('points','faceVertexCounts','faceVertexIndices','normals','primvars:displayColor','extent'):
                assert np.array_equal(np.asarray(l.GetPrim().GetAttribute(name).Get()),np.asarray(r.GetPrim().GetAttribute(name).Get())),name
        if prim.IsA(UsdGeom.Xformable):
            assert np.array_equal(np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()),np.asarray(UsdGeom.Xformable(right.GetPrimAtPath(prim.GetPath())).GetLocalTransformation()))


def package(visual,physics,output):
    output.mkdir(parents=True,exist_ok=False)
    v=Usd.Stage.Open(str(visual)); p=Usd.Stage.Open(str(physics)); assert v and p
    geometry_equal(p,v)
    before=physical_snapshot(p)
    assert p.GetDefaultPrim().GetAttribute('physics:mass').Get()>0
    # Transfer only the authored physics properties and APIs, preserving visual bindings/subsets.
    for prim in p.Traverse():
        path=prim.GetPath(); dest=v.GetPrimAtPath(path)
        if not dest:
            if 'PhysicsMaterialAPI' in prim.GetAppliedSchemas():
                Sdf.CopySpec(p.GetRootLayer(),path,v.GetRootLayer(),path)
            continue
        for schema in prim.GetAppliedSchemas():
            if 'Physics' in schema or 'Physx' in schema: dest.AddAppliedSchema(schema)
        for prop in prim.GetProperties():
            name=str(prop.GetName())
            if name.startswith(('physics:','physx','material:binding:physics')) and p.GetRootLayer().GetPropertyAtPath(prop.GetPath()):
                Sdf.CopySpec(p.GetRootLayer(),prop.GetPath(),v.GetRootLayer(),prop.GetPath())
    v.SetMetadata('kilogramsPerUnit',p.GetMetadata('kilogramsPerUnit'))
    assert physical_snapshot(v)==before,'Physics properties changed'
    # Flatten composition, then copy every asset-valued attribute to a content-addressed path.
    flat=v.Flatten(); packed=Usd.Stage.Open(flat)
    dependencies={}
    for prim in packed.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset,Sdf.ValueTypeNames.AssetArray): continue
            if attr.GetNumTimeSamples(): raise ValueError('Animated asset paths are not supported')
            value=attr.Get(); values=list(value) if attr.GetTypeName()==Sdf.ValueTypeNames.AssetArray else [value]
            replacements=[]
            for asset in values:
                if not asset or not asset.path: replacements.append(asset); continue
                if '<UDIM>' in asset.path: raise ValueError('UDIM dependency expansion is not implemented')
                path=Path(asset.resolvedPath or asset.path)
                if not path.is_absolute(): path=visual.parent/path
                path=path.resolve(); assert path.is_file(),path
                if path.suffix.lower() in ('.usd','.usda','.usdc','.usdz','.mdl'):
                    raise ValueError('Nested shader/layer asset dependency requires recursive packager: '+str(path))
                digest=sha(path); relative=Path('textures')/(digest[:16]+path.suffix.lower())
                (output/relative).parent.mkdir(exist_ok=True); shutil.copy2(path,output/relative)
                replacements.append(Sdf.AssetPath('./'+relative.as_posix()))
                dependencies[str(relative)]={'source':str(path),'sha256':digest}
            attr.Set(replacements if attr.GetTypeName()==Sdf.ValueTypeNames.AssetArray else replacements[0])
    flat.Export(str(output/'object.usda'))
    saved=Usd.Stage.Open(str(output/'object.usda')); geometry_equal(p,saved)
    assert physical_snapshot(saved)==before
    layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(output/'object.usda'))
    assert not unresolved,list(unresolved)
    for path in [*[x.realPath for x in layers],*assets]:
        assert Path(path).resolve().is_relative_to(output.resolve()),path
    report={'passed':True,'geometry_exact':True,'physics_exact':True,'physics_snapshot':before,
            'dependencies':dependencies,'asset_sha256':sha(output/'object.usda'),
            'dependency_closure_local':True,'source_visual_sha256':sha(visual),'source_physics_sha256':sha(physics)}
    (output/'package.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('visual',type=Path); parser.add_argument('physics',type=Path); parser.add_argument('output',type=Path)
    a=parser.parse_args(); package(a.visual.resolve(),a.physics.resolve(),a.output.resolve())
    print('APPEARANCE_PACKAGE_VERIFIED',flush=True)
