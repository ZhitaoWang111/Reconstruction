"""Decide, per generated object, which generator should make its asset: DualCraft or SAM3D.

DualCraft builds an articulated asset (links and joints) and is slow and costs money per
object; SAM3D makes one rigid mesh in seconds.  An object goes to DualCraft when it has a
rigid part a person moves on a joint: a hinged door or lid, a drawer, a flap.  Built-in
cabinets count.  Fabric, soft goods, wheels and decorative moving bits stay with SAM3D.

One model call per object (claude-sonnet-5-5 through the gateway in
``/home/seer/wzt/Gateway/key.json``): the object's label, its scanned size and up to three
key-frame crops go in; a JSON object comes back, held to ``ROUTE_SCHEMA`` by structured
outputs.  Besides the route the model names the moving parts it sees and which key frame
shows the object's front (the face a person uses it from), since the export needs the front
to place a DualCraft asset and the answer comes from the same look at the same images.

Results merge into ``<generation>/routing.json`` by object id, so ``--object-id`` reruns
keep the rest.  ``--dry-run`` prints the prompt and the images without calling anything.

A person can set a route by hand in ``<generation>/routing_overrides.json`` (the way
``sizes_overrides.json`` pins sizes)::

    {"obj_0049": {"route": "sam3d", "note": "..."}}

An override wins over the model: ``route`` is the route to use, ``model_route`` keeps the
model's answer and ``override`` the entry, and a model rerun of the object keeps it.
``--overrides-only`` reapplies the file to routing.json without calling the model.

Runs in the SceneAssembly venv, which has the Anthropic SDK::

    ../SceneAssembly/.venv/bin/python -m objgen.planning.route_generators output/scene0000_00/gt --dry-run
    ../SceneAssembly/.venv/bin/python -m objgen.planning.route_generators output/scene0000_00/gt --object-id obj_0038
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
from datetime import datetime
from pathlib import Path

import sys

import numpy as np
from PIL import Image

from ..config import ROOT
# The same frame rules and crop as the DualCraft inputs, so the router judges an object
# from the photos DualCraft would get.  Pure numpy / PIL / trimesh: runs in this venv.
from ..dualcraft import prepare_dualcraft_inputs as pdi  # noqa: E402
GATEWAY = Path("/home/seer/wzt/Gateway/key.json")
MODEL = "claude-sonnet-5-5"
SCHEMA = "routing/v1"
OVERRIDES = "routing_overrides.json"
ROUTES = ("dualcraft", "sam3d")
RULE = ("dualcraft when the object has a rigid part a person moves on a joint (door, lid, drawer, "
        "flap), built-in cabinets included; sam3d for fabric, soft goods, wheels, decorative "
        "moving bits and everything rigid without such parts")
IMAGE_SIDE = pdi.MAX_SIDE
CONTEXT_SIDE = 768      # the whole frame, shown small, so the model sees what the object stands against

MOTIONS = ["hinged_door", "drawer", "lid_or_flap", "sliding_door", "knob_or_dial", "other"]

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "identity": {
            "type": "string",
            "description": "What the object is, in a few words, as the photos show it.",
        },
        "label_matches": {
            "type": "boolean",
            "description": "Whether the scan label names this object correctly.",
        },
        "moving_parts": {
            "type": "array",
            "description": "Every rigid part that moves on a joint and that a person operates. "
                           "Empty when there is none.",
            "items": {
                "type": "object",
                "properties": {
                    "part": {"type": "string", "description": "e.g. 'left upper door', 'top drawer'"},
                    "motion": {"type": "string", "enum": MOTIONS},
                    "count": {"type": "integer", "description": "How many such parts."},
                    "visible": {"type": "boolean",
                                "description": "Seen in the photos, rather than assumed for this kind of object."},
                },
                "required": ["part", "motion", "count", "visible"],
                "additionalProperties": False,
            },
        },
        "route": {"type": "string", "enum": ["dualcraft", "sam3d"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string", "description": "One or two sentences."},
        "front": {
            "type": "object",
            "description": "The object's front: the face a person uses it from.",
            "properties": {
                "best_image": {
                    "type": "integer",
                    "description": "Index (from 1) of the image that shows the front most squarely; "
                                   "0 if no image shows the front.",
                },
                "facing": {
                    "type": "string",
                    "enum": ["toward_camera", "left", "right", "away_from_camera", "not_visible"],
                    "description": "In that image, where the front faces: at the viewer, toward the "
                                   "picture's left or right edge, or away.",
                },
                "cue": {"type": "string", "description": "What identifies the front (doors, drawer fronts, controls)."},
            },
            "required": ["best_image", "facing", "cue"],
            "additionalProperties": False,
        },
    },
    "required": ["identity", "label_matches", "moving_parts", "route", "confidence", "reasoning", "front"],
    "additionalProperties": False,
}

SYSTEM = """You sort the objects of a scanned indoor scene between two 3-D asset generators.

DualCraft builds an articulated model, a body with moving parts on joints, so a robot can \
open it. It takes minutes and costs money per object. SAM3D builds one rigid mesh in seconds.

Route an object to DualCraft when it has at least one rigid part that a person moves on a \
joint to use it: a hinged door or lid, a drawer, a flap, a sliding door. Built-in or \
wall-hung cabinets count; a cabinet door is a door wherever the cabinet is mounted. Route \
everything else to SAM3D: rigid objects without such parts, fabric and soft goods \
(curtains, pillows, bags, clothes), anything whose only moving parts are wheels, casters, \
handlebars or decorative bits, and parts too small to matter for opening the object (a \
toaster's lever, a keyboard tray).

Judge from the photos, not the label: scan labels are sometimes wrong or generic \
("object"). The photos are crops around the object from a video walked through the room, \
and other objects appear in them; the object is the one the crop centres on, inside the \
yellow box in the context frame. When you cannot see the moving parts but the object type \
always has them (a fridge, a microwave), list them with visible false. When the object is \
too unclear to judge, say so in the reasoning and give low confidence.

Then find the object's front: the face a person uses it from (doors, drawer fronts, \
controls, the display). Name the image that shows it most squarely and where the front \
faces in that image."""


def load_gateway() -> tuple[str, str]:
    data = json.loads(GATEWAY.read_text())
    return data["token"], data["Claude_URL"].rstrip("/")


def jpeg_block(image: Image.Image, side: int) -> dict:
    image = image.convert("RGB")
    image.thumbnail((side, side), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")}}


def build_content(handoff: Path, record: dict, camera: dict, floor_z: float | None) -> tuple[list, list, dict]:
    """The user message: text, the photos DualCraft would get, and the first of them whole
    for context.  Returns the content, what was shown, and the scan size and dropped frames."""
    import trimesh
    points = np.asarray(trimesh.load(handoff / record["cloud"]).vertices, dtype=np.float64)
    box = pdi.scan_box(points)
    centre = np.array([*box["centre_xy"], (box["bottom_z"] + box["top_z"]) / 2])
    cut = {k["seq"]: pdi.out_of_picture(points, k["camera_to_world"], camera) for k in record["key_frames"]}
    frames, dropped = pdi.select_frames(record["key_frames"], cut, centre)
    floor_note = ""
    if floor_z is not None:
        floor_note = f" Its lowest scanned point is {box['bottom_z'] - floor_z:.2f} m above the floor."
    count = len(frames)
    images = ("Image 1: a crop around the object" if count == 1 else
              f"Images 1-{count}: crops around the object from different key frames")
    text = (
        f"Scan label: {record.get('label') or 'unknown'}"
        f"{' (a generic label: the annotator did not name it)' if record.get('label_quality') == 'generic' else ''}.\n"
        f"Scanned size: about {box['long']:.2f} m x {box['short']:.2f} m footprint, "
        f"{box['height']:.2f} m tall (the scan sees only what the cameras saw, so a side against a "
        f"wall can read short).{floor_note}\n"
        f"{images}. Image {count + 1}: the first of them whole, the object inside the yellow box."
    )
    content: list[dict] = [{"type": "text", "text": text}]
    shown = []
    for index, frame in enumerate(frames, 1):
        mask = np.asarray(Image.open(handoff / frame["mask"])) > 0
        crop = pdi.crop_box(mask)
        content.append({"type": "text", "text": f"Image {index}:"})
        content.append(jpeg_block(Image.open(handoff / frame["rgb"]).crop(tuple(crop)), IMAGE_SIDE))
        shown.append({"image": index, "seq": frame["seq"], "role": frame["role"], "crop": crop,
                      "out_of_picture": round(cut[frame["seq"]], 3)})
    content.append({"type": "text", "text": f"Image {count + 1} (context):"})
    content.append(jpeg_block(Image.open(handoff / frames[0]["boxed"]), CONTEXT_SIDE))
    shown.append({"image": count + 1, "seq": frames[0]["seq"], "role": "context"})
    content.append({"type": "text", "text": "Answer with the JSON object."})
    size = {k: box[k] for k in ("height", "long", "short", "bottom_z", "points")}
    return content, shown, {"size": size, "dropped": dropped}


def call(client, content: list) -> tuple[dict, dict, str]:
    """One routing decision.  Structured output holds the reply to ROUTE_SCHEMA; thinking
    stays adaptive at low effort (a sorting judgement, not a long derivation)."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
        thinking={"type": "adaptive"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": ROUTE_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    usage = {"input": response.usage.input_tokens, "output": response.usage.output_tokens,
             "cache_creation": getattr(response.usage, "cache_creation_input_tokens", None),
             "cache_read": getattr(response.usage, "cache_read_input_tokens", None)}
    if response.stop_reason == "refusal":
        raise RuntimeError(f"refused: {getattr(response, 'stop_details', None)}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("reply cut off at max_tokens")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), usage, response.model


def front_seq(answer: dict, shown: list[dict]) -> int | None:
    """The key frame the model named for the front.  The context image is the first key
    frame uncropped, so naming it means that frame; 0 means no image shows a front."""
    best = answer["front"]["best_image"]
    image = next((s for s in shown if s["image"] == best), None)
    return image["seq"] if image is not None else None


def check(answer: dict, images: int) -> list[str]:
    """What the schema cannot say: indices in range, a route consistent with the parts.
    ``images`` counts the crops; the context image after them is a valid answer too."""
    problems = []
    route = answer.get("model_route") or answer["route"]     # the model's own answer, not an override
    best = answer["front"]["best_image"]
    if not 0 <= best <= images + 1:
        problems.append(f"front.best_image {best} is not an image index (1-{images + 1}, or 0)")
    operable = [p for p in answer["moving_parts"] if p["motion"] != "knob_or_dial"]
    if route == "dualcraft" and not answer["moving_parts"]:
        problems.append("route dualcraft with no moving part listed")
    if route == "sam3d" and operable and answer["confidence"] == "high":
        problems.append("route sam3d although a moving part is listed")
    return problems


def load_overrides(generation: Path) -> dict:
    """object_id -> {"route": ..., "note": ...} from ``<generation>/routing_overrides.json``."""
    path = generation / OVERRIDES
    if not path.is_file():
        return {}
    overrides = json.loads(path.read_text())
    bad = {oid: entry.get("route") for oid, entry in overrides.items() if entry.get("route") not in ROUTES}
    if bad:
        raise SystemExit(f"{path}: route must be one of {ROUTES}, got {bad}")
    return overrides


def apply_overrides(objects: list[dict], overrides: dict) -> list[str]:
    """Set every object's ``route`` to its override, else to the model's answer
    (``model_route``).  Returns the override ids that match no routed object."""
    for entry in objects:
        if "model_route" not in entry:          # fresh from the model, or written before overrides
            entry["model_route"] = entry.get("route")
        override = overrides.get(entry["object_id"])
        entry["route"] = override["route"] if override else entry["model_route"]
        entry["override"] = override
    return sorted(set(overrides) - {entry["object_id"] for entry in objects})


def write_routing(generation: Path, objects: list[dict]) -> dict:
    """Apply the overrides and write routing.json; the route counts."""
    unknown = apply_overrides(objects, load_overrides(generation))
    if unknown:
        print(f"WARNING: overrides for objects not in routing.json, ignored: {', '.join(unknown)}")
    objects = sorted(objects, key=lambda o: o["object_id"])
    counts = {}
    for entry in objects:
        counts[entry.get("route") or "error"] = counts.get(entry.get("route") or "error", 0) + 1
    (generation / "routing.json").write_text(json.dumps({
        "schema_version": SCHEMA, "generation": str(generation), "model": MODEL, "rule": RULE,
        "route_counts": counts,
        "overridden": [entry["object_id"] for entry in objects if entry.get("override")],
        "objects": objects,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False) + "\n")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generation", type=Path, help="generation directory (has manifest.json)")
    parser.add_argument("--object-id", action="append", dest="object_ids",
                        help="repeat to route a subset; default: every successfully generated object")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompt and the images for each object; no model call, nothing written")
    parser.add_argument("--overrides-only", action="store_true",
                        help=f"reapply {OVERRIDES} to routing.json; no model call")
    args = parser.parse_args()
    generation = args.generation.resolve()
    if args.overrides_only:
        path = generation / "routing.json"
        if not path.is_file():
            raise SystemExit(f"{path} not found: route the objects first")
        counts = write_routing(generation, json.loads(path.read_text())["objects"])
        overridden = json.loads(path.read_text())["overridden"]
        print(f"overrides applied ({', '.join(overridden) or 'none'}); totals {counts} -> {path}")
        return 0
    manifest = json.loads((generation / "manifest.json").read_text())
    handoff = Path(manifest["source_handoff"]["path"])
    handoff_manifest = json.loads((handoff / "manifest.json").read_text())
    floor_z = handoff_manifest.get("floor_z")
    camera = handoff_manifest["camera"]
    scene = {o["object_id"]: o for o in json.loads((handoff / "objects.json").read_text())}
    records = [r for r in json.loads((generation / "objects.json").read_text()) if r.get("status") == "success"]
    ids = [r["object_id"] for r in records]
    if args.object_ids:
        unknown = sorted(set(args.object_ids) - set(ids))
        if unknown:
            raise SystemExit("not a generated object: " + ", ".join(unknown))
        ids = [i for i in ids if i in args.object_ids]

    client = None
    if not args.dry_run:
        import anthropic
        token, base_url = load_gateway()
        client = anthropic.Anthropic(api_key=token, base_url=base_url, max_retries=4, timeout=300)

    results = []
    for number, oid in enumerate(ids, 1):
        record = scene[oid]
        content, shown, extra = build_content(handoff, record, camera, floor_z)
        images = sum(1 for s in shown if s["role"] != "context")
        if args.dry_run:
            text = "\n".join(b["text"] for b in content if b["type"] == "text")
            print(f"=== [{number}/{len(ids)}] {oid}\n{text}\n"
                  + "\n".join(f"  image {s['image']}: seq {s['seq']} {s['role']}" for s in shown)
                  + "".join(f"\n  not sent: seq {d['seq']} ({d['reason']})" for d in extra["dropped"]))
            continue
        started = time.time()
        entry = {"object_id": oid, "label": record.get("label"), "images": shown,
                 "frames_dropped": extra["dropped"], "scan_size_m": extra["size"]}
        try:
            answer, usage, served_by = call(client, content)
            entry.update(answer, usage=usage, model=served_by, problems=check(answer, images))
            entry["front"]["seq"] = front_seq(answer, shown)
        except Exception as error:
            entry.update(route=None, error=f"{type(error).__name__}: {error}")
        entry["seconds"] = round(time.time() - started, 1)
        results.append(entry)
        parts = ", ".join(f"{p['count']}x {p['motion']}" for p in entry.get("moving_parts", []) or []) or "-"
        print(f"[{number:2d}/{len(ids)}] {oid} {record.get('label') or '-':<16} -> {entry.get('route') or 'ERROR':<9} "
              f"{entry.get('confidence', '')[:1]:1} parts {parts:<28} front img {entry.get('front', {}).get('best_image', '-')} "
              f"{entry.get('error') or ''}{' PROBLEMS ' + '; '.join(entry['problems']) if entry.get('problems') else ''}",
              flush=True)
    if args.dry_run:
        print(f"\n{len(ids)} objects; system prompt:\n{SYSTEM}")
        return 0

    path = generation / "routing.json"
    previous = {}
    if path.is_file():
        previous = {o["object_id"]: o for o in json.loads(path.read_text()).get("objects", [])}
    previous.update({r["object_id"]: r for r in results})
    counts = write_routing(generation, list(previous.values()))
    tokens = {"input": sum(r.get("usage", {}).get("input", 0) for r in results),
              "output": sum(r.get("usage", {}).get("output", 0) for r in results)}
    print(f"\n{len(results)} routed this run ({tokens['input']} input / {tokens['output']} output tokens); "
          f"totals {counts} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
