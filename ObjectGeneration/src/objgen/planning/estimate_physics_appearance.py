"""Estimate physical and appearance properties for every generated object using a vision model.

For each successfully generated object the script sends the object label, scaled
dimensions and up to three key-frame images to claude-sonnet-5-5 and asks it to
fill a fixed schema with:

  mass_kg                  – estimated mass in kilograms
  roughness                – PBR roughness in [0, 1]
  metallic                 – PBR metallic in [0, 1]
  body_type                – "dynamic" | "kinematic" | "static"
  collision_approximation  – "convexHull" or "convexDecomposition"
  reasoning                – one-sentence justification, kept in the output

``kinematic`` is derived from ``body_type`` (anything other than "dynamic") so the
USD authoring stage keeps a single flag to branch on.

Objects whose asset came from DualCraft (``generator: dualcraft`` in objects.json)
take a narrower path: the asset already carries per-material roughness / metallic
and a per-link structure, so the model is asked only for the total mass and whether
the root body (the carcass, not the doors) is fixed.  The prompt adds the blueprint
part list, and DualCraft's own mass estimate is shown as a reference and recorded
next to ours (``mass_reference``).  Collision is fixed to per-piece convex hulls.

Results are written to <generation>/physics_appearance.json.  A second run merges
into the existing file by object_id; individual objects can be re-run with
--object-id without dropping the other records.

Usage::

    PYTHONPATH=src .venv-sam3d/bin/python -m objgen.planning.estimate_physics_appearance output/scene0000_00/gt

Gateway key file: /home/seer/wzt/Gateway/key.json
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..config import DUALCRAFT_REPO
from ..dualcraft import dualcraft_product as dp  # noqa: E402

GATEWAY = Path("/home/seer/wzt/Gateway/key.json")
MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 1024
MAX_RETRIES = 3
RETRY_DELAY = 5.0   # seconds between retries

TOOL = {
    "name": "set_properties",
    "description": (
        "Estimate physical and PBR-appearance properties of a single indoor object "
        "from its label, real-world dimensions, and one or more photographs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mass_kg": {
                "type": "number",
                "description": "Total mass in kilograms based on typical material and size."
            },
            "roughness": {
                "type": "number",
                "description": "PBR roughness 0.0 (mirror-smooth) to 1.0 (fully rough)."
            },
            "metallic": {
                "type": "number",
                "description": "PBR metallic 0.0 (non-metal) to 1.0 (pure metal)."
            },
            "body_type": {
                "type": "string",
                "enum": ["dynamic", "kinematic", "static"],
                "description": (
                    "How the object behaves physically.  Pick exactly one.\n"
                    "dynamic = a free body that gravity and collisions act on; a person "
                    "can lift or push it.  Anything merely RESTING on a floor, table, "
                    "counter or shelf is dynamic, even if heavy or hard to move "
                    "(toaster, microwave, trash can, bathroom scale, chair, bed).\n"
                    "kinematic = a body that stays where a script puts it but is meant "
                    "to be articulated: hinged or sliding door, drawer, curtain on a "
                    "rail, garage door.  Use this only when the object visibly moves as "
                    "a whole along a joint or track.\n"
                    "static = part of the building itself and never moves: built-in "
                    "kitchen counter, sink, shower enclosure, bathtub, wall-hung or "
                    "ceiling-hung cabinet, wall mirror, window, fixed railing."
                )
            },
            "collision_approximation": {
                "type": "string",
                "enum": ["convexHull", "convexDecomposition"],
                "description": (
                    "convexHull for box-like shapes (bed, cabinet, table, stool).  "
                    "convexDecomposition for concave or complex shapes (chair, "
                    "bicycle, guitar, shelf with open back)."
                )
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence justifying the estimates."
            }
        },
        "required": [
            "mass_kg", "roughness", "metallic",
            "body_type", "collision_approximation", "reasoning"
        ]
    }
}

SYSTEM = (
    "You are an indoor-scene physics and material estimator.  "
    "Given an object label, its real-world dimensions in metres, and photographs, "
    "estimate the properties requested.  Use common knowledge of household objects.  "
    "IMPORTANT: objects are hollow or frame-based structures, not solid blocks.  "
    "A wooden bed frame (hollow) is 30-60 kg, not 200 kg.  "
    "A fabric curtain pair is 2-8 kg.  A stool is 3-10 kg.  "
    "A bathroom mirror (glass + frame) is 3-15 kg.  "
    "A backpack (empty) is 0.5-2 kg.  A microwave is 10-18 kg.  "
    "A toilet (porcelain hollow) is 20-40 kg.  A guitar case is 2-6 kg.  "
    "A coffee table is 5-20 kg.  "
    "Always think: what material is this, is it hollow or solid, what is its real weight?\n\n"
    "For body_type, ask what the object is RESTING ON or ATTACHED TO.  "
    "Resting on something is not the same as being fixed to it: a toaster on a "
    "counter and a microwave in a cabinet niche are still dynamic, because a person "
    "can pick them up.  Only choose static when the object is part of the building "
    "and could not be removed without demolition.  Only choose kinematic when the "
    "object is designed to swing, slide or roll along a joint or track."
)

# Per-label mass ranges in kg.  Used to clamp obviously wrong model estimates.
MASS_RANGE: dict[str, tuple[float, float]] = {
    "backpack": (0.5, 3), "bed": (30, 80), "bicycle": (8, 20),
    "cabinet": (20, 100), "clock": (0.2, 3), "coffee table": (5, 30),
    "couch": (50, 150), "curtain": (1, 10), "desk": (20, 60),
    "dish rack": (0.5, 3), "door": (15, 50), "guitar": (1, 5),
    "guitar case": (2, 8), "kitchen cabinets": (20, 100),
    "kitchen counter": (40, 150), "laundry basket": (0.5, 3),
    "microwave": (8, 20), "mirror": (2, 20), "nightstand": (10, 40),
    "pillow": (0.3, 2), "refrigerator": (50, 150), "scale": (0.5, 5),
    "shelf": (10, 50), "shoes": (0.3, 2), "shower": (20, 80),
    "sink": (5, 30), "stool": (3, 15), "table": (10, 80),
    "tissue box": (0.1, 0.5), "toaster": (1, 3), "toaster oven": (3, 10),
    "toilet": (20, 50), "trash can": (0.5, 5), "tv": (5, 30),
}

FALLBACK = {
    "mass_kg": 10.0,
    "roughness": 0.6,
    "metallic": 0.0,
    "body_type": "dynamic",
    "kinematic": False,
    "collision_approximation": "convexHull",
    "reasoning": "fallback defaults",
    "confidence": "fallback",
}

# Articulated assets (DualCraft).  Appearance and collision come from the asset;
# joints carry the moving parts, so body_type is only about the root body.
ARTICULATED_TOOL = {
    "name": "set_properties",
    "description": (
        "Estimate the total mass of an articulated indoor object, and whether its main "
        "body is fixed to the building, from its label, size, part list and photographs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mass_kg": {
                "type": "number",
                "description": "Total mass in kilograms of the whole object: main body plus "
                               "every listed part (doors, drawers, shelves, handles).",
            },
            "body_type": {
                "type": "string",
                "enum": ["dynamic", "static"],
                "description": (
                    "The main body only; doors and drawers already move on their joints.\n"
                    "dynamic = the main body stands free: a person could push or tip it "
                    "(a free-standing cabinet, wardrobe, fridge, microwave on a counter).\n"
                    "static = the main body is part of the building or screwed to it: "
                    "built-in kitchen units, wall-hung cabinets, fitted wardrobes."
                ),
            },
            "reasoning": {"type": "string", "description": "One sentence justifying both."},
        },
        "required": ["mass_kg", "body_type", "reasoning"],
    },
}

ARTICULATED_SYSTEM = (
    "You are an indoor-scene physics estimator.  The object has been modelled as a main "
    "body with moving parts on joints; you are given its label, real-world size, part "
    "list and photographs.  Estimate the TOTAL mass, counting every listed part.  "
    "Objects are hollow or panel-built, not solid blocks: a tall wooden pantry cabinet "
    "is 30-90 kg, a kitchen wall cabinet 10-30 kg, a fridge 50-120 kg.  A second "
    "modelling system's estimate is shown as a reference; use your own judgement and "
    "do not copy it.\n\n"
    "For body_type, judge the main body only.  Choose static only when it is built into "
    "or fixed to the building; anything that merely stands on the floor is dynamic."
)


def _load_gateway() -> tuple[str, str]:
    data = json.loads(GATEWAY.read_text())
    return data["token"], data["Claude_URL"].rstrip("/")


def _image_content(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode()
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


def _call(token: str, base_url: str, messages: list[dict], tool: dict = TOOL) -> dict:
    """Call the API and return the tool-use input dict, or raise on error."""
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": "set_properties"},
        "messages": messages,
    }
    req = urllib.request.Request(
        base_url + "/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"transport/json error: {exc}") from exc

    tool_block = next(
        (b for b in body.get("content", []) if b.get("type") == "tool_use"), None
    )
    if tool_block is None:
        # Gateway sometimes routes to a thinking-enabled variant that returns
        # end_turn + text instead of tool_use.  Raise so the caller retries.
        has_thinking = any(b.get("type") == "thinking" for b in body.get("content", []))
        reason = "thinking-mode response (no tool_use)" if has_thinking else "no tool_use block in response"
        raise RuntimeError(reason)
    return tool_block["input"], body.get("usage", {})


def estimate_object(
    oid: str,
    label: str,
    dims_m: dict,
    kf_paths: list[Path],
    token: str,
    base_url: str,
    scale_source: str = "",
) -> dict:
    """Estimate properties for one object, with retries."""
    long = round(dims_m["footprint_long"], 3)
    short = round(dims_m["footprint_short"], 3)
    height = round(dims_m["height_m"], 3)
    dim_note = ""
    if scale_source and "footprint" in scale_source:
        dim_note = (
            "  Note: dimensions were estimated from the object's scan footprint "
            "only (height may be unreliable); rely primarily on the photographs "
            "and your knowledge of typical object sizes."
        )
    text = (
        f"{SYSTEM}\n\n"
        f"Object label: {label or 'unknown'}\n"
        f"Real-world dimensions (metres): "
        f"long side {long} × short side {short} × height {height}.{dim_note}\n"
        f"Using {len(kf_paths)} photograph(s).  "
        "Please estimate mass, roughness, metallic, kinematic, and "
        "collision_approximation."
    )
    content: list[dict] = [{"type": "text", "text": text}]
    for p in kf_paths:
        content.append(_image_content(p))

    messages = [{"role": "user", "content": content}]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            result, usage = _call(token, base_url, messages)
            # Clamp mass to label prior if the model estimate is out of range.
            lo, hi = MASS_RANGE.get(label or "", (0, 1e9))
            raw_mass = result.get("mass_kg", 0)
            if not (lo <= raw_mass <= hi):
                result["mass_kg"] = max(lo, min(hi, raw_mass))
                result["confidence"] = "model+clamped"
            else:
                result["confidence"] = "model"
            # Keep the model's justification so wrong classifications stay auditable.
            # body_type stays authoritative; kinematic is kept for USD authoring.
            result["kinematic"] = result.get("body_type") != "dynamic"
            result["usage"] = usage
            return result
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    print(f"  WARNING {oid}: {last_error} — using fallback defaults")
    return dict(FALLBACK)


def dualcraft_context(generation: Path, record: dict) -> dict:
    """Part list, DualCraft's mass estimate, and appearance read from the asset."""
    stage = (generation / record["product"]).parent
    prompt = json.loads((stage / "inputs" / "prompt.json").read_text())
    source = stage / "inputs" / "blueprint" / "source.json"
    estimates = json.loads(source.read_text()).get("estimates", {}) if source.exists() else {}
    product = dp.read_product(generation / record["product"])
    groups = dp.rigid_groups(product)
    movable = {j.child: j for j in product.joints if j.type != "fixed"}
    parts = []
    for part in prompt.get("parts", []):
        appearance = part.get("appearance") or {}
        joint = movable.get(part["id"])
        parts.append(f"- {part['id']} ({appearance.get('finish') or '?'}"
                     + (f", moves on a {joint.type} joint" if joint else "") + f"): {part.get('description', '')}")
    return {"parts": parts, "blueprint_mass_kg": estimates.get("mass_kg"),
            "blueprint_materials": estimates.get("materials"), "bodies": len(groups),
            "appearance": material_appearance(product)}


def material_appearance(product) -> dict:
    """Area-weighted roughness and metallic over the asset's MTL materials, for
    the summary record; the USD keeps every material as authored."""
    sys.path.insert(0, str(DUALCRAFT_REPO / "src"))
    from sim_export.urdf2usd import read_mtl, read_obj
    weights = {"roughness": 0.0, "metallic": 0.0, "area": 0.0}
    for link in product.links.values():
        for path, _ in link.visuals:
            obj = read_obj(path)
            materials = {}
            for ref in obj.mtllib:
                if (path.parent / ref).is_file():
                    materials.update(read_mtl(path.parent / ref))
            for group in obj.groups:
                material = materials.get(group.material or "")
                if material is None:
                    continue
                faces = float(group.faces)
                weights["roughness"] += material.roughness * faces
                weights["metallic"] += material.metallic * faces
                weights["area"] += faces
    if not weights["area"]:
        return {"roughness": None, "metallic": None}
    return {"roughness": round(weights["roughness"] / weights["area"], 3),
            "metallic": round(weights["metallic"] / weights["area"], 3),
            "note": "face-count-weighted mean over the asset's materials; each material keeps its own values"}


def print_dry_run(oid: str, label: str, dims_m: dict, kf_paths: list[Path], context: dict) -> None:
    """Show the articulated prompt exactly as it would be sent."""
    captured = {}

    def fake_call(token, base_url, messages, tool=TOOL):
        captured.update(messages=messages, tool=tool)
        raise _DryRun

    global _call
    real, _call = _call, fake_call
    try:
        estimate_articulated(oid, label, dims_m, kf_paths, context, None, None, retries=1)
    except _DryRun:
        pass
    finally:
        _call = real
    text = captured["messages"][0]["content"][0]["text"]
    print(f"(dualcraft, tool {captured['tool']['input_schema']['required']})\n" + text
          + "\nimages: " + ", ".join(str(p) for p in kf_paths)
          + f"\nappearance from asset: {context['appearance']}\n")


class _DryRun(Exception):
    pass


def estimate_articulated(oid: str, label: str, dims_m: dict, kf_paths: list[Path], context: dict,
                         token: str, base_url: str, retries: int = MAX_RETRIES) -> dict:
    """Total mass and root body type for a DualCraft asset, with retries."""
    reference = context.get("blueprint_mass_kg")
    text = (
        f"{ARTICULATED_SYSTEM}\n\n"
        f"Object label: {label or 'unknown'}\n"
        f"Real-world size at rest (metres): width {dims_m['width']:.3f} × depth "
        f"{dims_m['depth']:.3f} × height {dims_m['height']:.3f}.\n"
        f"Parts (id, finish, description):\n" + "\n".join(context["parts"]) + "\n"
        + (f"Materials named by the modelling system: {', '.join(context['blueprint_materials'])}.\n"
           if context.get("blueprint_materials") else "")
        + (f"Reference only, from the modelling system: {reference} kg.\n" if reference else "")
        + f"Using {len(kf_paths)} photograph(s).  Estimate mass_kg and body_type."
    )
    content: list[dict] = [{"type": "text", "text": text}]
    content += [_image_content(p) for p in kf_paths]
    messages = [{"role": "user", "content": content}]
    last_error = None
    for attempt in range(retries):
        try:
            result, usage = _call(token, base_url, messages, ARTICULATED_TOOL)
            lo, hi = MASS_RANGE.get(label or "", (0, 1e9))
            raw_mass = result.get("mass_kg", 0)
            result["confidence"] = "model"
            if not (lo <= raw_mass <= hi):
                result["mass_kg"] = max(lo, min(hi, raw_mass))
                result["confidence"] = "model+clamped"
            break
        except _DryRun:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
    else:
        print(f"  WARNING {oid}: {last_error} — using the DualCraft estimate")
        result = {"mass_kg": reference or FALLBACK["mass_kg"], "body_type": "dynamic",
                  "reasoning": "model unavailable; DualCraft blueprint estimate",
                  "confidence": "fallback"}
        usage = {}
    result.update(
        generator="dualcraft",
        roughness=context["appearance"]["roughness"], metallic=context["appearance"]["metallic"],
        appearance_source="asset materials (MTL)",
        collision_approximation="convexHullPerPiece",
        kinematic=False,
        mass_reference={"dualcraft_blueprint_kg": reference,
                        "ratio_to_reference": round(result["mass_kg"] / reference, 3) if reference else None},
        usage=usage,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "generation", type=Path, help="Generation output directory (has manifest.json)"
    )
    parser.add_argument(
        "--object-id", action="append", dest="object_ids",
        help="Repeat to process a subset; default is all successful objects",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print each prompt and the images it would send; no gateway call, nothing written",
    )
    args = parser.parse_args()
    generation = args.generation.resolve()

    manifest = json.loads((generation / "manifest.json").read_text())
    handoff = Path(manifest["source_handoff"]["path"])
    scene = {o["object_id"]: o for o in json.loads((handoff / "objects.json").read_text())}
    records = [
        r for r in json.loads((generation / "objects.json").read_text())
        if r["status"] == "success"
    ]
    if args.object_ids:
        unknown = sorted(set(args.object_ids) - {r["object_id"] for r in records})
        if unknown:
            raise SystemExit("No successful generation for: " + ", ".join(unknown))
        records = [r for r in records if r["object_id"] in args.object_ids]

    sizes = {
        s["object_id"]: s
        for s in json.loads((generation / "sizes.json").read_text()).get("objects", [])
    }

    token, base_url = (None, None) if args.dry_run else _load_gateway()

    results = []
    for i, record in enumerate(sorted(records, key=lambda r: r["object_id"]), 1):
        oid = record["object_id"]
        label = record.get("label") or ""
        src = scene.get(oid, {})
        size = sizes.get(oid, {})

        # Scaled dimensions in metres
        mn = size.get("mesh_normalised", {})
        sf = size.get("scale_factor", 1.0)
        dims_m = {
            "footprint_long": mn.get("footprint_long", 1.0) * sf,
            "footprint_short": mn.get("footprint_short", 1.0) * sf,
            "height_m": size.get("height_m", 0.5),
        }

        # Key-frame images (up to 3, primary first)
        kf_paths = []
        for kf in src.get("key_frames", []):
            p = handoff / kf["rgb"]
            if p.exists():
                kf_paths.append(p)
            if len(kf_paths) >= 3:
                break

        print(f"[{i:2d}/{len(records)}] {oid}  {label or '-':<20}  {len(kf_paths)} images", end="  ", flush=True)
        if record.get("generator") == "dualcraft":
            # DualCraft dims: width across the front (Y), depth front-to-back (X).
            context = dualcraft_context(generation, record)
            given = size.get("given_size_m") or {}
            asset = size.get("mesh_normalised", {})
            dims_m = {"width": asset.get("footprint_long", given.get("width", 0.0)),
                      "depth": asset.get("footprint_short", given.get("depth", 0.0)),
                      "height": asset.get("height", given.get("height", 0.0))}
            if args.dry_run:
                print_dry_run(oid, label, dims_m, kf_paths, context)
                continue
            result = estimate_articulated(oid, label, dims_m, kf_paths, context, token, base_url)
        else:
            if args.dry_run:
                print("(sam3d, unchanged prompt)")
                continue
            result = estimate_object(oid, label, dims_m, kf_paths, token, base_url,
                                     scale_source=size.get("source", ""))
        result["object_id"] = oid
        result["label"] = label
        results.append(result)
        usage = result.pop("usage", {})
        number = (lambda v: "-" if v is None else f"{v:.2f}")
        print(
            f"mass={result['mass_kg']:.1f}kg  "
            f"rough={number(result['roughness'])}  "
            f"metal={number(result['metallic'])}  "
            f"type={result['body_type']}  "
            f"{result['collision_approximation']}  "
            f"[in={usage.get('input_tokens',0)} out={usage.get('output_tokens',0)}]"
        )

    if args.dry_run:
        return
    out_path = generation / "physics_appearance.json"
    # Merge with existing results so --object-id re-runs don't drop other objects
    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            for r in prev.get("objects", []):
                existing[r["object_id"]] = r
        except (json.JSONDecodeError, KeyError):
            pass
    for r in results:
        existing[r["object_id"]] = r
    # Backfill body_type on records written before the enum existed, so the
    # merged file has one consistent schema rather than a mix.
    for r in existing.values():
        if "body_type" not in r:
            r["body_type"] = "kinematic" if r.get("kinematic") else "dynamic"
            r["body_type_backfilled"] = True
    # Apply mass clamp to all records including previously stored ones.
    for r in existing.values():
        if r.get("confidence") in ("model", "model+clamped"):
            lbl = r.get("label") or ""
            lo, hi = MASS_RANGE.get(lbl, (0, 1e9))
            raw = r.get("mass_kg", 0)
            if not (lo <= raw <= hi):
                r["mass_kg"] = round(max(lo, min(hi, raw)), 2)
                r["confidence"] = "model+clamped"
    merged = sorted(existing.values(), key=lambda r: r["object_id"])
    out = {
        "schema_version": "physics_appearance/v1",
        "generation": str(generation),
        "model": MODEL,
        "objects": merged,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    fallback_count = sum(1 for r in results if r.get("confidence") == "fallback")
    print(
        f"\n{len(results)} objects written to {out_path}"
        + (f"  ({fallback_count} fallback)" if fallback_count else "")
    )


if __name__ == "__main__":
    main()
