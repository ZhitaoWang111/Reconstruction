"""The structure USD layout: /Structure/Walls/<segment>, /Structure/Floors/<floor>.

Blender exports every top-level body flat under the root (/Structure/<name>); `regroup` moves
each one under Walls or Floors by the custom property it carries (handoff_wall / sg_floor).
Materials stay at /Structure/_materials, so bindings keep pointing where they did.
"""
from __future__ import annotations

from pxr import Sdf, Usd, UsdGeom

ROOT = "/Structure"
GROUPS = {"Walls": "userProperties:handoff_wall", "Floors": "userProperties:sg_floor"}


def group_of(prim) -> str | None:
    for group, attr in GROUPS.items():
        if prim.HasAttribute(attr):
            return group
    return None


def regroup(stage: Usd.Stage) -> dict[str, list[str]]:
    layer = stage.GetRootLayer()
    moved = {g: [] for g in GROUPS}
    root = stage.GetPrimAtPath(ROOT)
    plan = [(p.GetName(), group_of(p)) for p in root.GetChildren()]
    for group in GROUPS:
        UsdGeom.Xform.Define(stage, f"{ROOT}/{group}")
    edit = Sdf.BatchNamespaceEdit()
    for name, group in plan:
        if group:
            edit.Add(Sdf.NamespaceEdit.Reparent(f"{ROOT}/{name}", f"{ROOT}/{group}", -1))
            moved[group].append(name)
    if not layer.Apply(edit):
        raise RuntimeError("could not move bodies under Walls / Floors")
    return moved


def body_path(group: str, name: str) -> str:
    return f"{ROOT}/{group}/{name}/{name}_body"
