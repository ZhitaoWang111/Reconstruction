"""Read a built white model back and check it: USD structure and the plan's numbers.

    PYTHONPATH=src .venv-sam3d/bin/python -m structuregen.whitebox.validate <run dir>

Checks, against the run's plan.json:
* the USD holds one body mesh per segment, at /Structure/Walls/<segment>/<segment>_body, and
  one per floor at /Structure/Floors/<floor>/<floor>_body, with the named zones as material subsets;
* every floor's volume equals its top area x the plan's floor thickness;
* every body's volume from its mesh equals footprint x height minus its openings;
* every segment has the zones the plan implies (front and back always; reveal only where
  an opening is).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pxr import Usd, UsdGeom


def quad_volume(points, faces) -> float:
    """Signed volume of a closed triangle-fanned mesh, through the divergence theorem."""
    total = 0.0
    for f in faces:
        p0 = points[f[0]]
        for k in range(1, len(f) - 1):
            p1, p2 = points[f[k]], points[f[k + 1]]
            total += (p0[0] * (p1[1] * p2[2] - p2[1] * p1[2])
                      - p0[1] * (p1[0] * p2[2] - p2[0] * p1[2])
                      + p0[2] * (p1[0] * p2[1] - p2[0] * p1[1])) / 6.0
    return abs(total)


def jamb_shear(points) -> float:
    """How far the back face's inner columns sit from the front face's, along the wall.

    Segment-local frame: x along the wall, the front face at y = 0 and the back at y = -thickness.
    Every x the front has strictly inside the segment (opening jambs) must recur on the back;
    only the two end columns may differ, where a mitred corner slants the end face.
    """
    import numpy as np
    p = np.array(points, float)
    front = np.unique(np.round(p[np.abs(p[:, 1]) < 1e-5, 0], 5))
    back = np.unique(np.round(p[p[:, 1] < p[:, 1].min() + 1e-5, 0], 5))
    inner = front[1:-1]
    if not len(inner):
        return 0.0
    return float(max(np.abs(back - x).min() for x in inner))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)
    planned = json.loads((args.run / "plan.json").read_text())
    stage = Usd.Stage.Open(str(args.run / "structure_base.usda"))
    print("upAxis", UsdGeom.GetStageUpAxis(stage), "metersPerUnit", UsdGeom.GetStageMetersPerUnit(stage),
          "defaultPrim", stage.GetDefaultPrim().GetPath())

    found, floors = {}, {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if str(prim.GetParent().GetParent().GetPath()) == "/Structure/Floors":
            mesh = UsdGeom.Mesh(prim)
            counts = list(mesh.GetFaceVertexCountsAttr().Get())
            idx = list(mesh.GetFaceVertexIndicesAttr().Get())
            fs, k = [], 0
            for c in counts:
                fs.append(idx[k:k + c])
                k += c
            floors[prim.GetParent().GetName()] = quad_volume([tuple(p) for p in mesh.GetPointsAttr().Get()], fs)
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = [tuple(p) for p in mesh.GetPointsAttr().Get()]
        counts = list(mesh.GetFaceVertexCountsAttr().Get())
        idx = list(mesh.GetFaceVertexIndicesAttr().Get())
        faces, k = [], 0
        for c in counts:
            faces.append(idx[k:k + c])
            k += c
        seg = prim.GetParent().GetName()
        sheared = jamb_shear(pts)
        if sheared > 1e-4:
            found.setdefault("_sheared", []).append(f"{seg}: opening jambs not square to the wall "
                                                     f"(back edge {sheared:.3f} m off the front edge)")
        if str(prim.GetParent().GetParent().GetPath()) != "/Structure/Walls":
            found.setdefault("_bad_paths", []).append(str(prim.GetPath()))
            continue
        zones = sorted(sub.GetPrim().GetName().replace(seg + "_", "") for sub in UsdGeom.Subset.GetAllGeomSubsets(mesh))
        found[seg] = {"volume": quad_volume(pts, faces), "zones": zones,
                      "faces": len(faces), "path": str(prim.GetPath())}

    failures = []
    for seg in planned["segments"]:
        built = found.get(seg["id"])
        if not built:
            failures.append(f"{seg['id']}: no body mesh under /Walls")
            continue
        err = built["volume"] - seg["expected_volume_m3"]
        expect_zones = {"front", "back", "top", "bottom", "end_start", "end_end"}
        if seg["openings"]:
            expect_zones.add("reveal")
        missing = expect_zones - set(built["zones"])
        print(f"{seg['id']:9s} volume={built['volume']:.4f} expected={seg['expected_volume_m3']:.4f} "
              f"err={err:+.6f} faces={built['faces']} zones={','.join(built['zones'])}")
        if abs(err) > args.tolerance:
            failures.append(f"{seg['id']}: volume off by {err:+.6f}")
        if missing:
            failures.append(f"{seg['id']}: missing zones {sorted(missing)}")
    failures += found.pop("_sheared", [])
    for path in found.pop("_bad_paths", []):
        failures.append(f"{path}: body not at /Structure/Walls/<segment>/<segment>_body")
    extra = sorted(set(found) - {s["id"] for s in planned["segments"]})
    if extra:
        failures.append("unexpected bodies: " + ", ".join(extra))
    if str(stage.GetDefaultPrim().GetPath()) != "/Structure":
        failures.append("defaultPrim is not /Structure")
    t = planned.get("floor_thickness_m", 0.05)
    for f in planned.get("floors", []):
        v = floors.pop(f["id"], None)
        if v is None:
            failures.append(f"{f['id']}: no floor body under /Structure/Floors")
            continue
        print(f"{f['id']:20s} volume={v:.4f} expected={f['area_m2'] * t:.4f}")
        if abs(v - f["area_m2"] * t) > args.tolerance:
            failures.append(f"{f['id']}: floor volume {v:.5f}, area x thickness {f['area_m2'] * t:.5f}")
    if floors:
        failures.append("unexpected floors: " + ", ".join(sorted(floors)))

    for f in failures:
        print("VALIDATE_FAILURE", f, file=sys.stderr)
    print("STRUCTUREGEN_VALIDATE_OK" if not failures else "STRUCTUREGEN_VALIDATE_FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
