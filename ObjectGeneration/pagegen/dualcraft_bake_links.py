import trimesh, numpy as np, xml.etree.ElementTree as ET, json, sys
from pathlib import Path
prod, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(exist_ok=True)
r = ET.parse(prod / "model.urdf").getroot(); info = {}
for l in r.findall("link"):
    name = l.get("name")
    f = prod / l.find("visual").find("geometry")[0].get("filename")
    s = trimesh.load(f, process=False)
    geoms = list(s.geometry.values()) if isinstance(s, trimesh.Scene) else [s]
    parts = []
    for g in geoms:
        try:
            g.visual = g.visual.to_color()
        except Exception as e:
            print("colour fail", name, e)
        parts.append(g)
    m = trimesh.util.concatenate(parts)
    m.export(out / f"{name}.ply")
    info[name] = {"faces": len(m.faces), "pieces": len(geoms), "bounds": m.bounds.round(4).tolist()}
    print(name, len(geoms), len(m.faces), m.bounds.round(3).tolist())
json.dump(info, open(out / "info.json", "w"), indent=1)
