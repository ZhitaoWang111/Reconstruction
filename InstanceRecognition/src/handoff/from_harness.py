"""Write the object handoff (handoff/v3, amb3r_floor) from a finished harness run.

    python -m src.handoff.from_harness --run harness-v15-amb3r-point-full100frames-20261112-0935 \\
        --source artifacts/sources/scene0000_00/scene_amb3r_point_full100_v2_floor

The second producer `schema.py` was written for, beside `from_gt.py`; a consumer cannot
tell which one wrote a directory except from `manifest.producer`.

What comes from where
---------------------
* Which observations make which object: the run's last committed state - its live
  objects and the observations each one holds (at most five; the harness evicts the
  rest). Observations the harness discarded, excluded or evicted are not used.
* Geometry: the lifted points of those observations, read from `--source`. The run
  itself may have been made on an earlier lifting of the same masks (v15 ran on the
  scan-frame source); the source given here must be the same masks and pixels, which
  `match_sources` checks, and it must be in amb3r_floor. The association is a decision
  about which masks show one thing, so it carries over; the coordinates do not.
* Key frames: up to three of the held observations, chosen with the GT producer's rule
  (pixel ratio x clarity, then wider view separation), so the two producers' key
  frames mean the same thing. Mask = the observation's own mask.
* Camera: the scene's colour intrinsics (`<scene>.txt`, the same block as the GT and
  wall handoffs) and AMB3R's own pose per frame from the source - the pose the points
  were lifted with. Each key frame records `reprojection_px`: how far its own points land
  from their pixels through that pose and those intrinsics (AMB3R's depth and pose do not
  agree exactly; 5-17 px here, the range the wall handoff records too).

No ScanNet annotation is read. The scale of amb3r_floor is the one in the frame file
(`frame.scale`; oracle on ScanNet until a GT-free one exists).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from src.geometry.amb3r_mask_source import AMBR3R_INTERMEDIATE, AMBR3R_SIZE, POLICY_FLOOR
from src.geometry.backproject_masks import crop_box
from src.structure.frames import SCENE
from src.geometry.incremental_objects import voxelize

from . import schema
from .from_gt import draw_boxed, frame_clarity, read_scene_meta, write_ply, EDGE_PENALTY, MIN_FRAME_GAP, \
    SEPARATIONS_DEG, MAX_VIEWS, VISIBILITY_FLOOR, VISIBILITY_FLOOR_STRUCTURAL, select_keyframes

DEFAULT_RUN = "harness-v15-amb3r-point-full100frames-20261112-0935"
DEFAULT_SOURCE = config.ARTIFACTS_ROOT / "sources/scene0000_00/scene_amb3r_point_full100_v2_floor"
VOXEL = 0.02            # m; the harness's own fused-cloud voxel (run config `voxel`)
HELD = ("held", "assigned")   # 'assigned' is the pre-rename name for held


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- the run
def last_state(run_dir: Path) -> dict:
    ck = json.loads((run_dir / "checkpoint.json").read_text())
    return json.loads((run_dir / "versions" / f"{ck['version']:04d}" / "state.json").read_text())


def objects_of(state: dict) -> dict[str, dict]:
    """Live objects and the observations each holds: {object: {"label", "held": [obs ids]}}."""
    held: dict[str, list] = {}
    for oid, a in state["assignments"].items():
        if a.get("status") in HELD and a.get("object_id"):
            held.setdefault(a["object_id"], []).append(oid)
    out = {}
    for obj, info in state["objects"].items():
        if info.get("status") in ("merged", "retired") or not held.get(obj):
            continue
        out[obj] = {"label": info.get("object_label") or "", "held": sorted(held[obj])}
    return out


# ---------------------------------------------------------------- the source
class Source:
    """The lifted masks, by observation id `seq:instance` (the harness's own id)."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("lifting_policy") != POLICY_FLOOR or self.manifest.get("coordinate_frame") != schema.FRAME:
            raise SystemExit(f"{self.root} is not an {schema.FRAME} source (lifting_policy "
                             f"{self.manifest.get('lifting_policy')!r}); lift it with --frame amb3r_floor")
        self.frames = self.manifest["frames"]
        self._npz: dict[int, dict] = {}

    def frame(self, seq: int) -> dict:
        return self.frames[seq]

    def arrays(self, seq: int) -> dict:
        if seq not in self._npz:
            with np.load(self.root / self.frames[seq]["npz"]) as z:
                self._npz[seq] = {k: z[k] for k in ("points_world", "rgb", "mask_bits", "camera_to_world", "pixel_indices")}
        return self._npz[seq]

    def observation(self, obs_id: str) -> dict:
        seq, inst = (int(x) for x in obs_id.split(":"))
        frame = self.frames[seq]
        for bit, item in enumerate(frame["instances"]):
            if int(item["instance_id"]) == inst:
                z = self.arrays(seq)
                keep = ((z["mask_bits"] & np.uint32(1 << bit)) != 0) & np.isfinite(z["points_world"]).all(1)
                return {"seq": seq, "source_frame_id": int(frame["source_frame_id"]), "item": item,
                        "pixel_indices": z["pixel_indices"][keep],
                        "mask": (self.root / item["mask_file"]).resolve(),
                        "rgb": (self.root / frame["original_rgb_image"]).resolve(),
                        "points": z["points_world"][keep].astype(np.float64), "colors": z["rgb"][keep],
                        "camera_to_world": z["camera_to_world"].astype(np.float64)}
        raise KeyError(f"{obs_id}: no instance {inst} in source frame {seq}")


def match_sources(run_dir: Path, source: Source) -> dict:
    """The run's observations must be this source's masks and pixels, whatever frame the run's points were in."""
    obs = json.loads((run_dir / "observations.json").read_text())
    bad = []
    for oid, o in obs.items():
        seq = int(o["seq"])
        frame = source.frame(seq)
        item = next((i for i in frame["instances"] if int(i["instance_id"]) == int(o["instance_id"])), None)
        if item is None or int(frame["source_frame_id"]) != int(o["source_frame_id"]) \
                or Path(o["mask"]).resolve() != (source.root / item["mask_file"]).resolve():
            bad.append(oid); continue
        if len(source.observation(oid)["points"]) != int(o["point_count"]):
            bad.append(oid)
    if bad:
        raise SystemExit(f"{len(bad)} of the run's observations are not this source's masks/pixels "
                         f"(first: {bad[:5]}); give the source the run's masks were lifted from")
    return {"observations": len(obs), "checked": "frame id, mask file and point count of every observation"}


def reprojection_px(o: dict, K: np.ndarray, size: tuple[int, int]) -> float:
    """Median distance (original pixels) between where pose + K put an observation's points and their own pixels."""
    gw, gh = AMBR3R_SIZE; iw, ih = AMBR3R_INTERMEDIATE
    left, top, right, bottom = crop_box(AMBR3R_INTERMEDIATE, AMBR3R_SIZE)
    pix = o["pixel_indices"]; W, H = size
    gx = (left + ((pix % gw) + 0.5) * (right - left) / gw) * W / iw - 0.5
    gy = (top + ((pix // gw) + 0.5) * (bottom - top) / gh) * H / ih - 0.5
    T = o["camera_to_world"]; q = (o["points"] - T[:3, 3]) @ T[:3, :3]
    front = q[:, 2] > 1e-6
    u = K[0, 0] * q[front, 0] / q[front, 2] + K[0, 2]; v = K[1, 1] * q[front, 1] / q[front, 2] + K[1, 2]
    return float(np.median(np.hypot(u - gx[front], v - gy[front]))) if front.any() else float("nan")


# ---------------------------------------------------------------- one object
def candidates(obs: list[dict], clarity: dict, reference: float) -> list[dict]:
    """The GT producer's candidate records, one per held observation."""
    out = []
    for o in obs:
        m = np.asarray(Image.open(o["mask"]).convert("L")) > 0
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        h, w = m.shape
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        ratio = len(xs) / float(m.size)
        sharp = min(clarity[o["seq"]] / max(reference, 1e-6), 2.0)
        out.append({"seq": o["seq"], "source_frame_id": o["source_frame_id"], "pixels": int(len(xs)),
                    "ratio": ratio, "bbox": box, "fill": len(xs) / max(1.0, (box[2] - box[0]) * (box[3] - box[1])),
                    "touches_border": bool(box[0] == 0 or box[1] == 0 or box[2] == w or box[3] == h),
                    "score": ratio * (0.5 + 0.5 * sharp), "obs": o, "obs_id": o["id"], "mask_array": m})
    return out


def build(run: str, output: Path, source_root: Path = DEFAULT_SOURCE, scene: Path = SCENE) -> dict:
    run_dir = config.ARTIFACTS_ROOT / "experiments" / run
    meta = read_scene_meta(scene)
    camera = schema.camera(meta["colorWidth"], meta["colorHeight"], meta["fx_color"], meta["fy_color"],
                           meta["mx_color"], meta["my_color"],
                           source=f"{scene / (scene.name + '.txt')} (fx_color, fy_color, mx_color, my_color, "
                                  "colorWidth, colorHeight); the same intrinsics as the GT and wall handoffs")
    K = np.asarray(camera["K"], dtype=np.float64)
    size = (int(camera["width"]), int(camera["height"]))
    source = Source(source_root)
    provenance = match_sources(run_dir, source)
    state = last_state(run_dir)
    objects = objects_of(state)
    run_cfg = json.loads((run_dir / "config.json").read_text())
    output = Path(output).resolve()
    if output.exists():
        shutil.rmtree(output)
    (output / "clouds").mkdir(parents=True)
    (output / "keyframes").mkdir()

    seqs = sorted({int(o.split(":")[0]) for v in objects.values() for o in v["held"]})
    clarity = {s: frame_clarity(source.observation(next(o for v in objects.values() for o in v["held"]
                                                         if int(o.split(":")[0]) == s))["rgb"]) for s in seqs}
    reference = float(np.median(list(clarity.values())))
    records, n_by_label = [], {}
    for obj in sorted(objects, key=lambda k: int(k.removeprefix("Object")) if k.removeprefix("Object").isdigit() else k):
        info = objects[obj]
        obs = [{**source.observation(o), "id": o} for o in info["held"]]
        P = np.concatenate([o["points"] for o in obs]); C = np.concatenate([o["colors"] for o in obs])
        _, P, C = voxelize(P, C.astype(np.float32), VOXEL)
        C = np.clip(np.round(C), 0, 255).astype(np.uint8)
        name = f"obj_{int(obj.removeprefix('Object')):04d}" if obj.removeprefix("Object").isdigit() else obj
        write_ply(output / "clouds" / f"{name}.ply", P, C)
        centroid = P.mean(0)
        cands = candidates(obs, clarity, reference)
        for c in cands:
            d = c["obs"]["camera_to_world"][:3, 3] - centroid
            c["bearing"] = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else None
            c["camera_center"] = c["obs"]["camera_to_world"][:3, 3].tolist()
        quality = schema.label_quality(info["label"])
        chosen = select_keyframes(cands, structural=quality == "structural") if cands else []
        primary = chosen[0]["bearing"] if chosen else None
        frames = []
        for index, c in enumerate(chosen):
            d = output / "keyframes" / name; d.mkdir(parents=True, exist_ok=True)
            stem = f"{c['seq']:04d}"
            shutil.copyfile(c["obs"]["rgb"], d / f"{stem}_rgb.jpg")
            Image.fromarray(np.where(c["mask_array"], 255, 0).astype(np.uint8), "L").save(d / f"{stem}_mask.png")
            draw_boxed(Image.open(d / f"{stem}_rgb.jpg").convert("RGB"), c["bbox"], f"{name} {info['label']}").save(
                d / f"{stem}_boxed.jpg", quality=95, subsampling=0)
            angle = None
            if primary is not None and c["bearing"] is not None:
                angle = round(float(np.degrees(np.arccos(np.clip(float(c["bearing"] @ primary), -1, 1)))), 2)
            f = schema.empty_frame()
            f.update({"seq": c["seq"], "source_frame_id": c["source_frame_id"],
                      "rgb": f"keyframes/{name}/{stem}_rgb.jpg", "mask": f"keyframes/{name}/{stem}_mask.png",
                      "boxed": f"keyframes/{name}/{stem}_boxed.jpg", "bbox_xyxy": c["bbox"],
                      "visible_pixels": c["pixels"], "pixel_ratio": round(c["ratio"], 6), "bbox_fill": round(c["fill"], 4),
                      "clarity": round(clarity[c["seq"]], 3), "visibility": round(c.get("visibility", 1.0), 4),
                      "touches_border": c["touches_border"], "camera_center": c["camera_center"],
                      "camera_to_world": [[float(v) for v in row] for row in c["obs"]["camera_to_world"]],
                      "view_angle_deg": angle, "role": schema.KEYFRAME_ROLES[index],
                      "observation_id": c["obs_id"], "reprojection_px": round(reprojection_px(c["obs"], K, size), 1)})
            frames.append(f)
        r = schema.empty_object(name)
        r.update({"label": info["label"], "label_vocab": "harness", "label_quality": quality,
                  "structural": quality == "structural", "cloud": f"clouds/{name}.ply", "point_count": int(len(P)),
                  "centroid": [round(float(v), 5) for v in centroid],
                  "bounds": {"minimum": [round(float(v), 5) for v in P.min(0)],
                             "maximum": [round(float(v), 5) for v in P.max(0)]},
                  "frames_visible": len(info["held"]), "key_frames": frames})
        records.append(r)
        n_by_label[quality] = n_by_label.get(quality, 0) + 1

    m = source.manifest
    views = {}
    for r in records:
        views[len(r["key_frames"])] = views.get(len(r["key_frames"]), 0) + 1
    manifest = schema.manifest(
        m["scene_id"],
        producer={"name": "harness", "run": run, "run_dir": str(run_dir),
                  "note": "objects are the harness run's live objects and the observations each holds; geometry is "
                          "those observations' lifted points in amb3r_floor; no ScanNet annotation was read",
                  "sources": {"source": str(source.root), "source_manifest_sha256": sha256(source.root / "manifest.json"),
                              "run_source": run_cfg.get("source"), "run_state_version": state["version"],
                              "model": run_cfg.get("model"), "geometry_policy_of_run": run_cfg.get("geometry_policy")},
                  "source_match": provenance},
        frames=len(source.frames), objects=len(records), label_quality_counts=n_by_label, keyframe_view_counts=views,
        source_frame_ids=[int(f["source_frame_id"]) for f in source.frames], camera=camera,
        frame=m["frame"], floor_z=0.0,
        cloud_policy={"points": "union of the held observations' lifted points", "voxel_m": VOXEL,
                      "held_per_object_max": 5,
                      "not_used": "observations the harness discarded, excluded or evicted over the hold cap"},
        keyframe_policy=schema.keyframe_policy(
            pool="the object's held observations (at most five)", occlusion_floor=VISIBILITY_FLOOR,
            occlusion_floor_structural=VISIBILITY_FLOOR_STRUCTURAL, edge_penalty=EDGE_PENALTY,
            separation_deg=SEPARATIONS_DEG, max_views=MAX_VIEWS,
            min_frame_gap=MIN_FRAME_GAP))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "objects.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {output}: {len(records)} objects {n_by_label}, key-frame views {views}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=DEFAULT_RUN, help="harness run under artifacts/experiments")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="the amb3r_floor lifting of the run's masks")
    ap.add_argument("--output", type=Path, default=None,
                    help="default artifacts/handoff/<scene>/harness_<run short name>")
    a = ap.parse_args()
    scene = json.loads((Path(a.source) / "manifest.json").read_text())["scene_id"]
    short = a.run.split("-")[1] if a.run.startswith("harness-") else a.run
    build(a.run, a.output or config.ARTIFACTS_ROOT / "handoff" / scene / f"harness_{short}", a.source)


if __name__ == "__main__":
    main()
