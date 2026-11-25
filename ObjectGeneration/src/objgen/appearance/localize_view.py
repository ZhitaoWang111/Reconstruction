"""Rebase flattened Isaac view-scene textures into its adjacent texture folder."""
import sys
from pathlib import Path
from pxr import Usd,Sdf,UsdUtils
for raw in sys.argv[1:]:
    path=Path(raw).resolve();stage=Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName()==Sdf.ValueTypeNames.Asset:
                asset=attr.Get()
                if asset and asset.path:
                    local=path.parent/'textures'/Path(asset.path).name
                    assert local.is_file(),local
                    attr.Set(Sdf.AssetPath('./textures/'+local.name))
    stage.GetRootLayer().Save()
    layers,assets,missing=UsdUtils.ComputeAllDependencies(str(path));assert not missing
    assert all(Path(a).resolve().is_relative_to(path.parent) for a in assets)
