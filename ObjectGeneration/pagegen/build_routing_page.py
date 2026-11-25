"""Build the page for reviewing the generator routing (routing.json).

One object at a time: its representative key frame (the frame the model named as showing
the object's front, else the first key frame sent) beside what the model answered, with a
progress bar across the objects to step through them, and a table of every object below.

The frames are cut from the handoff's boxed key frames, so the object's box is drawn on
them (the model saw the same crops without the box), and are written as JPEG files under
``frames/`` beside the page; the page loads them as it steps, so it stays small.

Usage::

    .venv-sam3d/bin/python pagegen/build_routing_page.py output/scene0000_00/gt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "pagegen" / "routing" / "index.html"
DEFAULT_OUTPUT = ROOT / "site" / "routing" / "index.html"
SIDE = 900


def save_jpeg(image: Image.Image, path: Path, side: int = SIDE, quality: int = 80) -> None:
    image = image.convert("RGB")
    image.thumbnail((side, side), Image.LANCZOS)
    image.save(path, "JPEG", quality=quality)


def build(generation: Path, output: Path) -> None:
    routing = json.loads((generation / "routing.json").read_text())
    manifest = json.loads((generation / "manifest.json").read_text())
    handoff = Path(manifest["source_handoff"]["path"])
    scene = {o["object_id"]: o for o in json.loads((handoff / "objects.json").read_text())}
    frames_dir = output.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.jpg"):
        stale.unlink()

    objects = []
    for entry in routing["objects"]:
        oid = entry["object_id"]
        key_frames = {k["seq"]: k for k in scene[oid]["key_frames"]}
        images = []
        for shown in entry.get("images", []):
            if shown["role"] == "context":      # the first crop's frame whole: the "full" view below
                continue
            boxed = Image.open(handoff / key_frames[shown["seq"]]["boxed"])
            stem = f"{oid}_{shown['seq']:04d}"
            save_jpeg(boxed.crop(tuple(shown["crop"])), frames_dir / f"{stem}_crop.jpg")
            save_jpeg(boxed, frames_dir / f"{stem}_full.jpg")
            images.append({**{k: shown.get(k) for k in ("image", "seq", "role", "out_of_picture")},
                           "crop": f"frames/{stem}_crop.jpg", "full": f"frames/{stem}_full.jpg"})
        front_seq = (entry.get("front") or {}).get("seq")
        sent = [image["seq"] for image in images]
        representative = front_seq if front_seq in sent else (sent[0] if sent else None)
        objects.append({
            **{k: entry.get(k) for k in ("object_id", "label", "route", "model_route", "override",
                                         "confidence", "identity", "label_matches",
                                         "moving_parts", "reasoning", "front", "problems", "scan_size_m",
                                         "usage", "seconds", "model", "error")},
            "label_quality": scene[oid].get("label_quality"),
            "representative": {"seq": representative,
                               "from_front": front_seq is not None and representative == front_seq},
            "images": images,
            "dropped": entry.get("frames_dropped", []),
        })

    payload = {"generation": str(generation.relative_to(ROOT)) if generation.is_relative_to(ROOT) else str(generation),
               "model": routing.get("model"), "updated_at": routing.get("updated_at"), "objects": objects}
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", data))
    size = sum(p.stat().st_size for p in frames_dir.glob("*.jpg")) / 1e6
    print(f"wrote {output} ({output.stat().st_size / 1e3:.0f} KB) and {len(list(frames_dir.glob('*.jpg')))} "
          f"frames ({size:.1f} MB) for {len(objects)} objects")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generation", type=Path, help="generation directory (has routing.json)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.generation.resolve(), args.output)


if __name__ == "__main__":
    main()
