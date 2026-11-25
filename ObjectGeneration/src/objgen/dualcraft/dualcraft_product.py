"""Read a DualCraft product and rebuild its physics on our terms.

DualCraft hands over a URDF with one OBJ per link (plus MTL and baked textures) and
an appearance-only ``object.usdc``.  Its structure is trusted: link tree, joint
types, axes, origins and limits, and the per-material roughness / metallic.  Its
physics is not: masses are bounding-box volume x 400 kg/m^3, inertia is a solid
box, and every collision is the concave visual mesh itself.

This module keeps the structure and recomputes the physics:

* links hanging off a fixed joint (handles, knobs, shelves) are merged into the
  movable or root link they hang from, so physics sees one body per rigid group;
* each body's mass is split from the object's total by
  ``sum of convex-hull volumes x material density`` and its centre of mass and
  inertia are integrated over those hulls;
* collisions are the convex hull of every connected piece.  DualCraft builds
  links out of convex boards, so on obj_0038 and bath_cabinet every piece is
  already convex and the hull is exact.

Frames: the URDF root link frame is the DualCraft object frame, +X front,
+Y left, +Z up, footprint centred on the origin and standing on z = 0.

Nothing here talks to a model or to Isaac; it runs in the SAM3D venv.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

# kg/m^3 per material class.  Woods and boards follow scenesmith's materials.yaml
# (wood 600, plastic 950, glass 2500, rubber 1100, fabric 500); there is no brass
# entry, so metal fittings use its steel value.
DENSITY_KG_M3 = {
    "wood": 600.0,
    "metal": 7800.0,
    "plastic": 950.0,
    "glass": 2500.0,
    "rubber": 1100.0,
    "fabric": 500.0,
}
# DualCraft blueprint finishes (perception/common/blueprint/schema.py FINISH_VALUES)
# mapped to a material class.  matte / glossy say nothing about the material; the
# object's blueprint materials decide, and plastic is the fallback.
FINISH_TO_MATERIAL = {
    "wood": "wood",
    "brushed metal": "metal",
    "polished metal": "metal",
    "transparent glass": "glass",
    "rubber": "rubber",
    "fabric": "fabric",
}
# Plausible bulk density of a material class as built, used to flag a total mass
# that does not fit the geometry.  Panel furniture is board over air gaps, so wood
# reaches well below solid timber.
DENSITY_RANGE_KG_M3 = {
    "wood": (350.0, 900.0),
    "metal": (2000.0, 9000.0),
    "plastic": (300.0, 1500.0),
    "glass": (1500.0, 2800.0),
    "rubber": (500.0, 1500.0),
    "fabric": (50.0, 1000.0),
}
MIN_BODY_MASS_KG = 0.02
# Pieces smaller than this hold no volume worth simulating (stray triangles,
# zero-thickness trims); they stay visible but get no collision and no mass.
MIN_PIECE_VOLUME_M3 = 1e-9
ROOT_FRAME = ("DualCraft object frame: +X front, +Y left, +Z up; footprint bounding box "
              "centred on the origin, standing on z=0; metres")


def _floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.array([float(v) for v in text.split()] if text else default, dtype=np.float64)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw, column vectors: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_matrix(element: ET.Element | None) -> np.ndarray:
    """4x4, column vectors: p_parent = T @ p_child."""
    matrix = np.eye(4)
    if element is None:
        return matrix
    matrix[:3, :3] = rpy_matrix(_floats(element.get("rpy"), (0.0, 0.0, 0.0)))
    matrix[:3, 3] = _floats(element.get("xyz"), (0.0, 0.0, 0.0))
    return matrix


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray                     # 4x4, child frame in parent frame at q = 0
    axis: np.ndarray                       # unit vector in the child frame
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None


@dataclass
class Link:
    name: str
    visuals: list[tuple[Path, np.ndarray]] = field(default_factory=list)  # (obj, 4x4 in link frame)


@dataclass
class Product:
    directory: Path
    robot: str
    links: dict[str, Link]
    joints: list[Joint]
    root: str

    def parent_joint(self, link: str) -> Joint | None:
        return next((j for j in self.joints if j.child == link), None)

    def link_in_root(self, link: str) -> np.ndarray:
        """4x4 of a link frame in the root frame at q = 0."""
        matrix = np.eye(4)
        while (joint := self.parent_joint(link)) is not None:
            matrix = joint.origin @ matrix
            link = joint.parent
        return matrix


def read_product(directory: Path) -> Product:
    directory = Path(directory)
    urdf = directory / "model.urdf"
    root = ET.parse(urdf).getroot()
    links = {}
    for element in root.findall("link"):
        link = Link(element.get("name"))
        for visual in element.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                raise ValueError(f"{urdf}: link {link.name} has a non-mesh visual; DualCraft "
                                 "products are expected to be OBJ meshes only")
            scale = _floats(mesh.get("scale"), (1.0, 1.0, 1.0))
            transform = origin_matrix(visual.find("origin")) @ np.diag([*scale, 1.0])
            link.visuals.append(((directory / mesh.get("filename")).resolve(), transform))
        links[link.name] = link
    joints = []
    for element in root.findall("joint"):
        limit = element.find("limit")
        axis = _floats(element.find("axis").get("xyz") if element.find("axis") is not None else None,
                       (1.0, 0.0, 0.0))
        number = (lambda key: float(limit.get(key)) if limit is not None and limit.get(key) else None)
        joints.append(Joint(
            name=element.get("name"), type=element.get("type", "fixed"),
            parent=element.find("parent").get("link"), child=element.find("child").get("link"),
            origin=origin_matrix(element.find("origin")), axis=axis / (np.linalg.norm(axis) or 1.0),
            lower=number("lower"), upper=number("upper"),
            effort=number("effort"), velocity=number("velocity")))
    children = {j.child for j in joints}
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(f"{urdf}: expected one root link, found {roots}")
    unsupported = sorted({j.type for j in joints} - {"fixed", "revolute", "prismatic", "continuous"})
    if unsupported:
        raise ValueError(f"{urdf}: unsupported joint types {unsupported}")
    return Product(directory, root.get("name") or directory.name, links, joints, roots[0])


def load_link_mesh(link: Link) -> trimesh.Trimesh:
    """All visuals of a link, in the link frame, welded by position."""
    parts = []
    for path, transform in link.visuals:
        loaded = trimesh.load(path, process=False)
        geometries = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
        for geometry in geometries:
            mesh = trimesh.Trimesh(vertices=np.asarray(geometry.vertices, dtype=np.float64),
                                   faces=np.asarray(geometry.faces), process=False)
            mesh.apply_transform(transform)
            parts.append(mesh)
    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def piece_hulls(mesh: trimesh.Trimesh) -> tuple[list[trimesh.Trimesh], dict]:
    """Convex hull of every connected piece, plus how convex the pieces were.

    ``convexity`` is piece volume / hull volume over watertight pieces: 1.0 means the
    hull is exact.  Degenerate pieces (fewer than four distinct points, or no volume)
    are dropped and counted.
    """
    hulls, ratios, dropped = [], [], 0
    for piece in mesh.split(only_watertight=False):
        if len(np.unique(np.round(piece.vertices, 7), axis=0)) < 4:
            dropped += 1
            continue
        try:
            hull = piece.convex_hull
        except Exception:           # qhull rejects coplanar input
            dropped += 1
            continue
        if hull.volume < MIN_PIECE_VOLUME_M3:
            dropped += 1
            continue
        hulls.append(hull)
        if piece.is_watertight:
            ratios.append(abs(piece.volume) / hull.volume)
    return hulls, {
        "pieces": len(hulls),
        "dropped_degenerate": dropped,
        "convexity_min": round(min(ratios), 4) if ratios else None,
        "convexity_median": round(float(np.median(ratios)), 4) if ratios else None,
    }


def rigid_groups(product: Product) -> dict[str, list[str]]:
    """Body name -> links it absorbs.  A body is the root or the child of a movable
    joint; every link reached from it through fixed joints joins it."""
    groups: dict[str, list[str]] = {}

    def body_of(link: str) -> str:
        while (joint := product.parent_joint(link)) is not None and joint.type == "fixed":
            link = joint.parent
        return link

    for name in product.links:
        groups.setdefault(body_of(name), []).append(name)
    order = list(product.links)
    return {body: sorted(members, key=order.index) for body, members in groups.items()}


def body_in_root(product: Product, body: str) -> np.ndarray:
    return product.link_in_root(body)


def rpy_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Inverse of rpy_matrix: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2, 0])))
    if abs(rotation[2, 0]) < 1 - 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:                           # gimbal lock: fold yaw into roll
        roll, yaw = math.atan2(-rotation[1, 2], rotation[1, 1]), 0.0
    return np.array([roll, pitch, yaw])


def translation(offset: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


def link_in_body(product: Product, link: str, body: str) -> np.ndarray:
    return np.linalg.inv(body_in_root(product, body)) @ product.link_in_root(link)


def read_blueprint(run: Path) -> dict:
    """The blueprint DualCraft drafted: part list with finishes, and its estimates."""
    prompt = json.loads((run / "inputs" / "prompt.json").read_text())
    source_path = run / "inputs" / "blueprint" / "source.json"
    estimates = json.loads(source_path.read_text()).get("estimates", {}) if source_path.exists() else {}
    return {"text": prompt.get("text"), "parts": prompt.get("parts", []), "estimates": estimates}


def link_materials(product: Product, blueprint: dict) -> dict[str, dict]:
    """Material class per link, from the blueprint part finish when the part exists."""
    object_materials = " ".join(blueprint.get("estimates", {}).get("materials") or []).lower()
    fallback = "wood" if "wood" in object_materials else "plastic"
    parts = {part["id"]: part for part in blueprint.get("parts", [])}
    result = {}
    for name in product.links:
        finish = (parts.get(name, {}).get("appearance") or {}).get("finish")
        material = FINISH_TO_MATERIAL.get(finish or "", fallback)
        result[name] = {"finish": finish, "material": material,
                        "source": "blueprint finish" if finish in FINISH_TO_MATERIAL else
                                  f"object materials ({fallback})" if finish else "no blueprint part"}
    return result


def axis_provenance(trajectory: Path | None) -> dict[str, dict]:
    """Per joint: did the agent settle the axis (pose_check / set_joint_axis) or is it
    the placeholder.  Read from the harness's joint_axis_set events."""
    if trajectory is None or not trajectory.exists():
        return {}
    if trajectory.suffix == ".zst":
        # The SAM3D venv has no zstandard module; the zstd CLI is enough here.
        import subprocess
        text = subprocess.run(["zstd", "-dc", str(trajectory)], check=True,
                              capture_output=True).stdout.decode()
    else:
        text = trajectory.read_text()
    result: dict[str, dict] = {}
    for line in text.splitlines():
        if '"joint_axis_set"' not in line:
            continue
        event = json.loads(line)
        for name, joint in (event.get("joints") or {}).items():
            result[name] = {"turn": event.get("turn"), "source": joint.get("source"),
                            "changed": bool(joint.get("changed")), "was": joint.get("was")}
    return result


@dataclass
class Body:
    name: str
    links: list[str]
    material_volumes: dict[str, float]     # material class -> hull volume, m^3
    hulls: list[trimesh.Trimesh]           # in the body frame
    hull_material: list[str]
    pieces: dict[str, dict]                # link -> piece_hulls stats
    mass: float = 0.0
    centre_of_mass: np.ndarray | None = None
    inertia: np.ndarray | None = None      # 3x3 about the centre of mass, body axes


def build_bodies(product: Product, materials: dict[str, dict]) -> dict[str, Body]:
    bodies = {}
    for body, members in rigid_groups(product).items():
        volumes: dict[str, float] = {}
        hulls, hull_material, pieces = [], [], {}
        for link in members:
            mesh = load_link_mesh(product.links[link])
            mesh.apply_transform(link_in_body(product, link, body))
            link_hulls, stats = piece_hulls(mesh)
            material = materials[link]["material"]
            volumes[material] = volumes.get(material, 0.0) + sum(h.volume for h in link_hulls)
            hulls += link_hulls
            hull_material += [material] * len(link_hulls)
            pieces[link] = stats
        bodies[body] = Body(body, members, volumes, hulls, hull_material, pieces)
    return bodies


def distribute_mass(bodies: dict[str, Body], total_kg: float) -> dict:
    """Split ``total_kg`` by hull volume x material density, then integrate each
    body's centre of mass and inertia over its hulls at that density.

    One factor scales every density so the bodies add up to the total; that factor
    times a class density is the implied bulk density reported for checking.
    """
    raw = {name: sum(DENSITY_KG_M3[m] * v for m, v in body.material_volumes.items())
           for name, body in bodies.items()}
    raw_total = sum(raw.values())
    if raw_total <= 0:
        raise ValueError("no closed volume to distribute mass over")
    factor = total_kg / raw_total
    for name, body in bodies.items():
        body.mass = max(MIN_BODY_MASS_KG, raw[name] * factor)
        # Integrate each hull at its own density, then combine about the body COM.
        scale = body.mass / raw[name] if raw[name] > 0 else 0.0
        masses = [DENSITY_KG_M3[m] * h.volume * scale for h, m in zip(body.hulls, body.hull_material)]
        if not body.hulls or sum(masses) <= 0:
            body.centre_of_mass = np.zeros(3)
            body.inertia = np.eye(3) * 1e-6
            continue
        centres = np.array([h.center_mass for h in body.hulls])
        com = (centres * np.array(masses)[:, None]).sum(0) / sum(masses)
        inertia = np.zeros((3, 3))
        for hull, mass, centre in zip(body.hulls, masses, centres):
            local = hull.moment_inertia / hull.mass * mass if hull.mass > 0 else np.zeros((3, 3))
            d = centre - com
            inertia += local + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        body.centre_of_mass, body.inertia = com, inertia
    total_volume = {m: sum(b.material_volumes.get(m, 0.0) for b in bodies.values())
                    for m in DENSITY_KG_M3}
    implied = {m: round(DENSITY_KG_M3[m] * factor, 1) for m, v in total_volume.items() if v > 0}
    return {"density_factor": round(factor, 4), "implied_density_kg_m3": implied,
            "hull_volume_m3": {m: round(v, 5) for m, v in total_volume.items() if v > 0}}


def principal(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal moments and a proper rotation whose columns are the principal axes."""
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    return np.maximum(moments, 1e-9), axes


def rest_bounds(product: Product) -> np.ndarray:
    """Axis-aligned bounds of the closed object in the root frame, 2x3."""
    lows, highs = [], []
    for name, link in product.links.items():
        mesh = load_link_mesh(link)
        mesh.apply_transform(product.link_in_root(name))
        lows.append(mesh.bounds[0])
        highs.append(mesh.bounds[1])
    return np.array([np.min(lows, axis=0), np.max(highs, axis=0)])


def asset_offset(bounds: np.ndarray) -> np.ndarray:
    """Where the asset origin sits in the DualCraft frame: bounding-box XY centre,
    lowest point.  Assets share one convention (XY box centred, lowest point at
    z = 0) whichever generator made them, so ``p_dualcraft = p_asset + offset``."""
    return np.array([(bounds[0, 0] + bounds[1, 0]) / 2, (bounds[0, 1] + bounds[1, 1]) / 2,
                     bounds[0, 2]])


def joint_motion(joint: Joint, value: float) -> np.ndarray:
    """Child-frame motion at joint position ``value`` (radians or metres)."""
    motion = np.eye(4)
    if joint.type in ("revolute", "continuous"):
        x, y, z = joint.axis
        c, s, t = math.cos(value), math.sin(value), 1 - math.cos(value)
        motion[:3, :3] = [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                          [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                          [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]
    elif joint.type == "prismatic":
        motion[:3, 3] = joint.axis * value
    return motion


def _halfspaces(hull: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(hull.face_normals)
    offsets = np.einsum("ij,ij->i", normals, hull.triangles[:, 0])
    return normals, offsets


def hulls_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh, depth: float) -> bool:
    """True when two convex hulls interpenetrate deeper than ``depth``: both, shrunk
    by ``depth`` along every face normal, still share a point (a linear program)."""
    if np.any(a.bounds[0] + depth > b.bounds[1] - depth) or np.any(b.bounds[0] + depth > a.bounds[1] - depth):
        return False
    from scipy.optimize import linprog
    na, oa = _halfspaces(a)
    nb, ob = _halfspaces(b)
    result = linprog(np.zeros(3), A_ub=np.vstack([na, nb]),
                     b_ub=np.concatenate([oa - depth, ob - depth]),
                     bounds=[(None, None)] * 3, method="highs")
    return result.status == 0


def _placed_hulls(product: Product, bodies: dict[str, "Body"], values: dict[str, float]) -> dict:
    placed = {}
    for name, body in bodies.items():
        matrix, link = np.eye(4), name
        while (joint := product.parent_joint(link)) is not None:
            matrix = joint.origin @ joint_motion(joint, values.get(joint.name, 0.0)) @ matrix
            link = joint.parent
        placed[name] = [h.copy().apply_transform(matrix) for h in body.hulls]
    return placed


def _extent(hulls: list) -> np.ndarray:
    lows = np.min([h.bounds[0] for h in hulls], axis=0)
    highs = np.max([h.bounds[1] for h in hulls], axis=0)
    return highs - lows


def interference(product: Product, bodies: dict[str, "Body"], depth: float = 0.002,
                 sweep_depth: float = 0.01) -> dict:
    """Geometry problems PhysX will not sort out by itself.

    ``rest_overlaps``: bodies with no joint between them that interpenetrate at rest.
    PhysX collides those, so it would shove them apart on the first step.

    ``swept_into_parent``: a movable body that, at a joint limit, sits inside the body
    it is jointed to.  PhysX never collides a joint's two bodies, so nothing stops a
    door hinged on the wrong side from swinging through its cabinet; only the limits
    bound it.  This usually means the axis sign or the limits are wrong.

    ``moves_inward``: the same mistake caught by direction rather than overlap.  A
    door hinged at the very edge of its carcass swings inward flush against the side
    wall and barely overlaps anything, so the overlap test can miss it.  Opening must
    carry the moving body away from the parent across the plane the body closes: its
    centre must not end up behind its closed position, on the parent's side.  The
    plane normal points from the parent's centre to the body's closed centre.
    """
    group_of = {link: body for body, body_ in bodies.items() for link in body_.links}
    joined = {frozenset((group_of[j.parent], group_of[j.child])) for j in product.joints
              if j.type != "fixed"}
    rest = _placed_hulls(product, bodies, {})
    names = sorted(rest)
    rest_overlaps = [[a, b] for i, a in enumerate(names) for b in names[i + 1:]
                     if frozenset((a, b)) not in joined
                     and any(hulls_overlap(ha, hb, depth) for ha in rest[a] for hb in rest[b])]
    swept = []
    for joint in product.joints:
        if joint.type not in ("revolute", "prismatic"):
            continue
        child, parent = group_of[joint.child], group_of[joint.parent]
        for bound in (joint.lower, joint.upper):
            if bound is None or abs(bound) < 1e-9:
                continue
            placed = _placed_hulls(product, {child: bodies[child], parent: bodies[parent]},
                                   {joint.name: bound})
            if any(hulls_overlap(hc, hp, sweep_depth) for hc in placed[child] for hp in placed[parent]):
                swept.append({"joint": joint.name, "at": round(bound, 4), "into": parent})
    inward = []
    parent_centre = {name: np.mean([h.center_mass for h in hulls], axis=0) for name, hulls in rest.items()}
    for joint in product.joints:
        if joint.type not in ("revolute", "prismatic"):
            continue
        child, parent = group_of[joint.child], group_of[joint.parent]
        for bound in (joint.lower, joint.upper):
            if bound is None or abs(bound) < 1e-9:
                continue
            moved = _placed_hulls(product, {child: bodies[child]}, {joint.name: bound})[child]
            before = np.mean([h.center_mass for h in rest[child]], axis=0)
            after = np.mean([h.center_mass for h in moved], axis=0)
            # Outward normal of the face the closed body covers, from the parent's
            # centre to the body's closed centre along the parent's dominant axis
            # (a front door's normal is +X, a lid's +Z).
            offset = before - parent_centre[parent]
            axis = int(np.argmax(np.abs(offset) / np.maximum(_extent(rest[parent]), 1e-9)))
            normal = np.zeros(3)
            normal[axis] = np.sign(offset[axis]) or 1.0
            travel = float(np.dot(after - before, normal))
            if travel < -0.01:
                inward.append({"joint": joint.name, "at": round(bound, 4), "axis": "XYZ"[axis],
                               "travel_m": round(travel, 4)})
    return {"rest_overlaps": rest_overlaps, "swept_into_parent": swept, "moves_inward": inward,
            "depth_m": depth, "sweep_depth_m": sweep_depth}
