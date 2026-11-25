"""Run in the USD environment; export authoritative arrays for baseline Blender."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Vt


def extract(source, output):
    stage = Usd.Stage.Open(str(source))
    root = stage.GetDefaultPrim()
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(root.GetPath().AppendChild('Mesh')))
    assert mesh and UsdGeom.GetStageUpAxis(stage) == 'Z'
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert len([p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]) == 1
    for prim in (root, mesh.GetPrim()):
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        assert np.allclose(np.asarray(matrix), np.eye(4), rtol=0, atol=1e-12), 'Non-canonical transform'
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype='<f4')
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype='<i4')
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype='<i4')
    colors = np.asarray(mesh.GetDisplayColorPrimvar().Get(), dtype='<f4')
    assert np.all(counts == 3) and len(colors) == len(points)
    assert np.isfinite(points).all() and np.isfinite(colors).all()
    valid, reason = UsdGeom.Mesh.ValidateTopology(Vt.IntArray.FromNumpy(indices), Vt.IntArray.FromNumpy(counts), len(points))
    assert valid, reason
    arrays = {'points': points, 'counts': counts, 'indices': indices, 'colors': colors}
    np.savez_compressed(output / 'canonical_mesh.npz', **arrays)
    fingerprints = {name: hashlib.sha256(arr.tobytes()).hexdigest() for name, arr in arrays.items()}
    report = {'schema_version': 1, 'default_prim': str(root.GetPath()), 'mesh_path': str(mesh.GetPath()),
              'up_axis': 'Z', 'meters_per_unit': 1.0, 'vertices': len(points), 'faces': len(counts),
              'dimensions_m': np.ptp(points, axis=0).tolist(), 'array_sha256': fingerprints,
              'geometry_sha256': hashlib.sha256((fingerprints['points'] + fingerprints['counts'] + fingerprints['indices']).encode()).hexdigest()}
    (output / 'geometry.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    extract(args.source, args.output)
