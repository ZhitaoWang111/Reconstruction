"""The GT structure page's data: what the ScanNet annotation says about walls, doors and windows.

    python -m pagegen.build_structure_gt            # -> docs/assets/structure-gt/data.js (window.STRUCT)

For docs/structure-gt-3d.html, and as the GT reference that pagegen.build_structure_geom
draws for comparison. Ground truth for evaluation only: nothing in src/ reads it.

From the annotated mesh (`_vh_clean_2.ply`, segments, aggregation):

1. Faces of the mesh are sampled (4000 points per m2) and every structural instance is split
   into planes by sequential RANSAC.
2. Wall planes that face the same way (within 10 deg) and lie within 15 cm of each other are
   one wall face. Faces are named F1, F2, ... by area, largest first.
3. Every wall vertex goes to the face whose plane it lies closest to (within 10 cm, its normal
   agreeing); each face is refitted and measured: rectangle, area, height, tilt, the step
   between two instances that make one face.
4. Doors, door frames, windows, mirrors and curtains are expressed on the face they hang on:
   width, height, offset from the wall plane.
5. scene0000_00 only, because they name its instances: the bathroom doorway inferred from
   door frame #63, and the four thicknesses that can be measured (two partition faces seen
   from both sides; the entry door set back in the east wall).

Coordinates are ScanNet's axis-aligned frame, for viewing; the comparison page moves them
into its own floor frame.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from src import config

SCENE = config.SCANNET_ROOT / "scene0000_00"
DEFAULT_OUT = config.ROOT / "docs/assets/structure-gt/data.js"
STRUCT = ("wall", "floor", "ceiling", "window", "door", "doorframe", "mirror", "curtain")
ITEM_LABELS = ("door", "doorframe", "window", "mirror", "curtain")
SAMPLES_PER_M2 = 4000
Z = np.array([0, 0, 1.0])

# scene0000_00: names for the faces and instances, the doorway to infer and the thicknesses to measure
NAMES = {
    "scene0000_00": {
        "faces": {"F1": "南墙", "F2": "东墙", "F3": "北墙（东半段）", "F4": "卫生间北隔墙，外侧面", "F5": "卫生间西隔墙直段，外侧面",
                  "F6": "西墙", "F7": "卫生间西隔墙斜段，外侧面", "F8": "卫生间西隔墙斜段，内侧面", "F9": "卫生间西隔墙直段，内侧面"},
        "items": {54: "入户门，门扇", 65: "入户门，门框", 63: "卫生间门框", 0: "北墙的窗", 1: "卫生间的窗", 64: "镜子",
                  5: "西墙的窗帘", 6: "南墙的窗帘"},
        "item_order": (54, 65, 63, 0, 1, 64, 5, 6),
        "doorway": {"frame": 63, "face": "F4"},
    },
}


# ---------------------------------------------------------------- the annotated mesh
def read_mesh(path: Path):
    with open(path, "rb") as f:
        n = nf = None
        while True:
            t = f.readline().decode().split()
            if t[:2] == ["element", "vertex"]:
                n = int(t[2])
            if t[:2] == ["element", "face"]:
                nf = int(t[2])
            if t == ["end_header"]:
                break
        v = np.fromfile(f, dtype=[("p", "<f4", 3), ("c", "u1", 4)], count=n)
        fa = np.fromfile(f, dtype=[("k", "u1"), ("i", "<i4", 3)], count=nf)
    return v["p"].astype(np.float64), v["c"][:, :3], fa["i"]


def load(scene: Path):
    P, C, F = read_mesh(scene / f"{scene.name}_vh_clean_2.ply")
    seg = np.asarray(json.loads((scene / f"{scene.name}_vh_clean_2.0.010000.segs.json").read_text())["segIndices"])
    agg = json.loads((scene / f"{scene.name}.aggregation.json").read_text())
    own = {s: g["objectId"] for g in agg["segGroups"] for s in g["segments"]}
    owner = np.array([own.get(int(s), -1) for s in seg])
    labels = {g["objectId"]: g["label"] for g in agg["segGroups"]}
    A = None
    for line in (scene / f"{scene.name}.txt").read_text().splitlines():
        if line.startswith("axisAlignment"):
            A = np.array([float(x) for x in line.split("=")[1].split()]).reshape(4, 4)
    return P, C, F, owner, labels, A


def vertex_normals(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    out = np.zeros_like(P)
    for k in range(3):
        np.add.at(out, F[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- 1. samples and planes
def sample_faces(P, F, owner, rng, per_m2=SAMPLES_PER_M2):
    """Points on the mesh faces; a face belongs to an instance when two of its vertices do."""
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2
    n = n / np.maximum(2 * area[:, None], 1e-12)
    o = owner[F]
    fo = np.where(o[:, 0] == o[:, 1], o[:, 0], np.where(o[:, 1] == o[:, 2], o[:, 1], np.where(o[:, 0] == o[:, 2], o[:, 0], -2)))
    k = rng.poisson(area * per_m2)
    idx = np.repeat(np.arange(len(F)), k)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
    return pts, n[idx], fo[idx], area, fo


def ransac(pts, rng, tol=0.03, iters=400):
    best, best_n = None, 0
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); length = np.linalg.norm(n)
        if length < 1e-9:
            continue
        n /= length
        inl = np.abs((pts - s[0]) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    q = pts[best]
    c = q.mean(0)
    _, v = np.linalg.eigh(np.cov((q - c).T))
    n = v[:, 0]
    return c, n, np.abs((pts - c) @ n) < tol


def instance_planes(pts, nrm, fo, area, face_owner, labels, rng):
    planes = []
    for oid, lab in sorted(labels.items()):
        if lab not in STRUCT:
            continue
        q, nq = pts[fo == oid], nrm[fo == oid]
        left = np.ones(len(q), bool)
        for k in range(5):
            if left.sum() < max(300, 0.06 * len(q)):
                break
            c, n, inl = ransac(q[left], rng)
            sel = np.flatnonzero(left)[inl]
            if len(sel) < max(300, 0.06 * len(q)):
                break
            if nq[sel].mean(0) @ n < 0:                  # the side the surface faces
                n = -n
            e = q[sel]
            planes.append({"object": oid, "label": lab, "k": k, "share": len(sel) / len(q), "c": c.tolist(), "n": n.tolist(),
                           "lo": e.min(0).tolist(), "hi": e.max(0).tolist(),
                           "area_m2": float(area[face_owner == oid].sum() * len(sel) / len(q))})
            left[sel] = False
    return planes


# ---------------------------------------------------------------- 2. wall faces
def wall_faces(planes):
    """Wall planes on one line facing one way are one face; named F1.. by area, largest first."""
    groups = []
    for p in (p for p in planes if p["label"] == "wall" and p["share"] >= 0.3):
        n = np.array(p["n"]); n[2] = 0; n /= np.linalg.norm(n)
        c = np.array(p["c"])
        for g in groups:
            if g["n"] @ n > np.cos(np.radians(10)) and abs((c - g["c"]) @ g["n"]) < 0.15:
                g["members"].append(p); break
        else:
            groups.append({"n": n, "c": c, "members": [p]})
    groups.sort(key=lambda g: -sum(m["area_m2"] for m in g["members"]))
    return [{"name": f"F{i + 1}", "ids": sorted({m["object"] for m in g["members"]}),
             "n": g["n"].round(3).tolist(), "point": g["c"].round(3).tolist()} for i, g in enumerate(groups)]


def pca(q):
    c = q.mean(0)
    _, v = np.linalg.eigh(np.cov((q - c).T))
    return c, v[:, 0]


def build(scene: Path = SCENE, out: Path = DEFAULT_OUT) -> dict:
    rng = np.random.default_rng(0)
    P0, C, F, owner, labels, A = load(scene)
    P = P0 @ A[:3, :3].T + A[:3, 3]
    names = NAMES.get(scene.name, {"faces": {}, "items": {}, "item_order": None, "doorway": None})

    # ---- 1. samples and planes (planes on the float64 samples, measurements on float32 ones, as first built)
    pts, nrm, fo, area_f, face_owner = sample_faces(P, F, owner, rng)
    planes = instance_planes(pts, nrm, fo, area_f, face_owner, labels, rng)
    SP, SO = pts.astype(np.float32).astype(float), fo.astype(np.int32)

    # ---- 2-3. wall faces: every wall vertex to one face, refit, measure
    faces = wall_faces(planes)
    VN = vertex_normals(P, F)
    tri_area = np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1) / 2
    face_of = np.full(len(P), -1, int)
    for f in faces:
        n, c = np.array(f["n"]), np.array(f["point"])
        f["planes"] = [p for p in planes if p["label"] == "wall" and p["share"] >= 0.3 and p["object"] in f["ids"]
                       and (lambda m: m / np.linalg.norm(m))(np.array(p["n"][:2] + [0])) @ n > np.cos(np.radians(10))
                       and abs((np.array(p["c"]) - c) @ n) < 0.15]
    best = np.full(len(P), 0.10)
    for fi, f in enumerate(faces):
        for p in f["planes"]:
            c, n = np.array(p["c"]), np.array(p["n"])
            d = np.abs((P - c) @ n)
            ok = (owner == p["object"]) & (VN @ n > 0.3) & (d < best)
            face_of[ok] = fi
            best[ok] = d[ok]
    wall_ids = [i for i, l in labels.items() if l == "wall"]
    is_wall = np.isin(owner, wall_ids)

    out_faces = []
    for fi, f in enumerate(faces):
        q = P[face_of == fi]
        c, n = pca(q)
        if VN[face_of == fi].mean(0) @ n < 0:
            n = -n
        d = (q - c) @ n
        nh = np.array([n[0], n[1], 0]); nh /= np.linalg.norm(nh)
        u = np.cross(Z, nh)                              # along the wall, horizontal
        v = np.cross(n, u)                               # up the fitted plane
        a, b = (q - c) @ u, (q - c) @ v
        alo, ahi = np.percentile(a, [0.5, 99.5]); blo, bhi = np.percentile(b, [0.5, 99.5])
        corners = [c + x * u + y * v for x, y in ((alo, blo), (ahi, blo), (ahi, bhi), (alo, bhi))]
        area = tri_area[(face_of[F] == fi).sum(1) >= 2].sum()
        per_id = {i: float(np.median((P[(face_of == fi) & (owner == i)] - c) @ n)) for i in f["ids"]}
        out_faces.append({
            "key": f["name"], "name": names["faces"].get(f["name"], f["name"]), "ids": f["ids"],
            "c": c.tolist(), "n": n.tolist(), "nh": nh.tolist(), "u": u.tolist(),
            "corners": [k.tolist() for k in corners], "anchor": (c + 0.03 * n).tolist(),
            "area_m2": round(float(area), 2), "rect_m2": round(float((ahi - alo) * (bhi - blo)), 2),
            "length_m": round(float(ahi - alo), 2), "z": [round(float(q[:, 2].min()), 2), round(float(np.percentile(q[:, 2], 99.5)), 2)],
            "rms_cm": round(float(np.sqrt((d ** 2).mean()) * 100), 1), "tilt_deg": round(float(np.degrees(np.arcsin(abs(n[2])))), 1),
            "offset_between_ids_cm": round(float(max(per_id.values()) - min(per_id.values())) * 100, 1) if len(per_id) > 1 else None,
            "a": [float(alo), float(ahi)]})
    print(f"{len(out_faces)} wall faces; {(face_of >= 0).sum() / is_wall.sum():.1%} of {is_wall.sum()} wall vertices on a face")

    # ---- 4. things on the walls, each on its host face
    def host_of(q):
        c, n = pca(q)
        best_d, arg = 1e9, None
        for fi, f in enumerate(out_faces):
            fn, fc, fu = np.array(f["nh"]), np.array(f["c"]), np.array(f["u"])
            if abs(n @ fn) < 0.85:
                continue
            a = (c - fc) @ fu
            if not (f["a"][0] - 0.5 < a < f["a"][1] + 0.5):
                continue
            dd = abs((c - fc) @ np.array(f["n"]))
            if dd < best_d:
                best_d, arg = dd, fi
        return arg

    def on_face(q, fi, lo=1, hi=99):
        f = out_faces[fi]
        fc, fu, n3 = np.array(f["c"]), np.array(f["u"]), np.array(f["n"])
        return np.percentile((q - fc) @ fu, [lo, hi]), np.percentile(q[:, 2], [lo, hi]), float(np.median((q - fc) @ n3))

    def rect3d(fi, ar, zr, s, lift=0.03):
        """Rectangle on the fitted (possibly tilted) plane of face fi, pushed s + lift along its normal."""
        f = out_faces[fi]
        fc, fu, n3 = np.array(f["c"]), np.array(f["u"]), np.array(f["n"])
        fv = np.cross(n3, fu)
        res = []
        for a, z in ((ar[0], zr[0]), (ar[1], zr[0]), (ar[1], zr[1]), (ar[0], zr[1])):
            b = (z - fc[2] - (s + lift) * n3[2]) / fv[2]
            res.append((fc + a * fu + b * fv + (s + lift) * n3).tolist())
        return res

    order = names["item_order"] or tuple(sorted(i for i, l in labels.items() if l in ITEM_LABELS))
    items = []
    for oid in order:
        q = SP[SO == oid]
        fi = host_of(q)
        if fi is None:
            continue
        ar, zr, s = on_face(q, fi)
        area = tri_area[(owner[F] == oid).sum(1) >= 2].sum()
        items.append({"key": f"{labels[oid]}#{oid}", "label": labels[oid], "object": oid,
                      "name": names["items"].get(oid, f"{labels[oid]}#{oid}"), "host": out_faces[fi]["key"],
                      "width_m": round(float(ar[1] - ar[0]), 2), "height_m": round(float(zr[1] - zr[0]), 2),
                      "z": [round(float(zr[0]), 2), round(float(zr[1]), 2)], "offset_cm": round(s * 100, 1),
                      "area_m2": round(float(area), 2), "fill": round(float(area / ((ar[1] - ar[0]) * (zr[1] - zr[0]))), 2),
                      "corners": rect3d(fi, ar, zr, s), "anchor": rect3d(fi, [ar.mean()] * 2, [zr.mean()] * 2, s)[0],
                      "inferred": False})

    # ---- 5. scene-specific: the doorway inside a door frame, and the measurable thicknesses
    gaps = []
    key_fi = {f["key"]: i for i, f in enumerate(out_faces)}
    dw = names.get("doorway")
    if dw:
        fi = key_fi[dw["face"]]
        f = out_faces[fi]
        fc, fu, n3 = np.array(f["c"]), np.array(f["u"]), np.array(f["n"])
        q = SP[SO == dw["frame"]]
        a, z = (q - fc) @ fu, q[:, 2]
        mid = (np.percentile(a, 1) + np.percentile(a, 99)) / 2
        body = (z > 0.2) & (z < 1.6)
        left, right = np.percentile(a[body & (a < mid)], 97), np.percentile(a[body & (a > mid)], 3)
        head = np.percentile(z[(a > left + 0.1) & (a < right - 0.1) & (z > 1.5)], 3)
        sa, ss, sz = (SP - fc) @ fu, (SP - fc) @ n3, SP[:, 2]
        inside = (sa > left + 0.05) & (sa < right - 0.05) & (sz > 0.10) & (sz < head - 0.05) & (np.abs(ss + 0.05) < 0.15)
        s_frame = [i for i in items if i["object"] == dw["frame"]][0]["offset_cm"] / 100
        items.append({"key": f"opening@{dw['frame']}", "label": "opening", "object": None, "name": "卫生间门洞（推算）",
                      "host": dw["face"], "width_m": round(float(right - left), 2), "height_m": round(float(head), 2),
                      "z": [0.0, round(float(head), 2)], "offset_cm": None, "area_m2": None, "fill": None,
                      "mesh_inside_m2": round(float(inside.sum() / SAMPLES_PER_M2), 3),
                      "corners": rect3d(fi, [left, right], [0.0, head], s_frame / 2, lift=0.0),
                      "anchor": rect3d(fi, [(left + right) / 2] * 2, [1.0, 1.0], s_frame / 2, lift=0.0)[0], "inferred": True})

        def gap(name, key_a, pts_b, label_b, zmid=1.0):
            fa = out_faces[key_fi[key_a]]
            ca, na, ua = np.array(fa["c"]), np.array(fa["n"]), np.array(fa["u"])
            d = (pts_b - ca) @ na
            along = (pts_b - ca) @ ua
            lo, hi = max(fa["a"][0], np.percentile(along, 2)), min(fa["a"][1], np.percentile(along, 98))
            amid = (lo + hi) / 2 if hi > lo else float(np.median(along))
            med = float(np.median(d))
            p = ca + amid * ua; p[2] = zmid
            p = p - ((p - ca) @ na) * na
            return {"name": name, "a": key_a, "b": label_b, "cm": round(abs(med) * 100, 1),
                    "spread_cm": round(float(np.percentile(d, 75) - np.percentile(d, 25)) * 100, 1),
                    "overlap_m": round(float(max(0, hi - lo)), 2), "p0": p.tolist(), "p1": (p + med * na).tolist()}

        fv = lambda k: P[face_of == key_fi[k]]
        f4 = out_faces[fi]
        gaps = [gap("卫生间西隔墙，直段厚度", "F5", fv("F9"), "F9"),
                gap("卫生间西隔墙，斜段厚度", "F7", fv("F8"), "F8"),
                gap("卫生间北隔墙厚度（外侧面到门框内侧面）", "F4",
                    SP[(SO == 63) & (np.abs((SP - np.array(f4["c"])) @ np.array(f4["n"]) + 0.1) < 0.04)], "doorframe#63"),
                gap("入户门扇比东墙面凹进", "F2", SP[SO == 54], "door#54")]

    # ---- floor, ceiling, areas per category
    floor_ids = [i for i, l in labels.items() if l == "floor"]
    ceiling_ids = [i for i, l in labels.items() if l == "ceiling"]
    fl = P[np.isin(owner, floor_ids)]
    floor = {"z_median": round(float(np.median(fl[:, 2])), 3),
             "area_m2": round(float(tri_area[(np.isin(owner, floor_ids)[F]).sum(1) >= 2].sum()), 1),
             "planes": [{"share": round(p["share"], 3), "z": [round(p["lo"][2], 2), round(p["hi"][2], 2)]}
                        for p in planes if p["label"] == "floor"]}
    ceiling = {"area_m2": round(float(tri_area[(np.isin(owner, ceiling_ids)[F]).sum(1) >= 2].sum()), 1),
               "planes": [{"share": round(p["share"], 3), "slope_deg": round(90 - float(np.degrees(np.arcsin(abs(p["n"][2])))), 1),
                           "z": [round(p["lo"][2], 2), round(p["hi"][2], 2)], "y": [round(p["lo"][1], 2), round(p["hi"][1], 2)]}
                          for p in planes if p["label"] == "ceiling"]}
    cat_area = {}
    for l in STRUCT:
        ids = [i for i, x in labels.items() if x == l]
        cat_area[l] = round(float(tri_area[np.isin(owner, ids)[F].sum(1) >= 2].sum()), 1)
    cat_area["unannotated"] = round(float(tri_area[(owner[F] < 0).sum(1) >= 2].sum()), 1)
    cat_area["total"] = round(float(tri_area.sum()), 1)
    cat_area["other"] = round(cat_area["total"] - sum(v for k, v in cat_area.items() if k != "total"), 1)

    # ---- pack
    lo, hi = P.min(0), P.max(0)
    q16 = np.round((P - lo) / (hi - lo) * 65535).astype("<u2")
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    meta = {"scene": scene.name, "vertices": len(P), "triangles": len(F), "lo": lo.tolist(), "hi": hi.tolist(),
            "labels": {str(k): v for k, v in labels.items()}, "faces": out_faces, "items": items, "gaps": gaps,
            "floor": floor, "ceiling": ceiling, "cat_area": cat_area,
            "wall_vertices_on_faces": round(float((face_of >= 0).sum() / is_wall.sum()), 3)}
    js = ("window.STRUCT = " + json.dumps(meta, ensure_ascii=False) + ";\n"
          + f'window.STRUCT.pos = "{b64(q16)}";\nwindow.STRUCT.idx = "{b64(F.astype("<u4"))}";\n'
          + f'window.STRUCT.rgb = "{b64(C.astype(np.uint8))}";\nwindow.STRUCT.owner = "{b64((owner + 1).astype(np.uint8))}";\n'
          + f'window.STRUCT.face = "{b64(np.where(face_of < 0, 255, face_of).astype(np.uint8))}";\n')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(js)
    print(f"{len(items)} items, {len(gaps)} thicknesses -> {out} ({len(js) / 1e6:.1f} MB)")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    build(a.scene, a.out)


if __name__ == "__main__":
    main()
