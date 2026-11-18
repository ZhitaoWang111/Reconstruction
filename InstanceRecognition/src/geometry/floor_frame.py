"""The delivery frame `amb3r_floor`: AMB3R's own reconstruction, scaled and stood upright.

    python -m src.geometry.floor_frame --run struct_v4_win5      # -> artifacts/scenes/<scene>/amb3r_floor.json

Definition: docs/amb3r-floor-frame.md. In short - origin on the floor below the first
camera, +Z the normal of the floor we segmented (the side the cameras are on), +X the first
camera's optical axis projected onto the floor, metres. One file per scene; every handoff
and every stage of the wall pipeline reads it instead of fitting its own.

Nothing from the ScanNet annotation is read. The one exception is the scale when
`scale_source` is `oracle_gt_poses`: the Umeyama scale between AMB3R's and ScanNet's
camera centres, a stand-in until a GT-free scale exists. Only the scale is taken from it.

`scannet_scan_to_frame` maps ScanNet's scan frame into this one. It is for evaluation and
for the GT producer only; nothing that builds a harness result may call it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import config

SCENE_ID = "scene0000_00"
FRAME_NAME = "amb3r_floor"
DEFINITION = "docs/amb3r-floor-frame.md"
TOL = 0.03              # m; RANSAC inlier band for the floor plane
SUBSAMPLE = 80000       # floor points the RANSAC samples from
ITERS = 400
SEED = 0
MAX_PITCH = 0.95        # |forward . up| above this: the frame looks straight down/up, use the next one
SCALE_SOURCES = ("oracle_gt_poses", "native")


def frame_path(scene_id: str = SCENE_ID) -> Path:
    return config.ARTIFACTS_ROOT / "scenes" / scene_id / f"{FRAME_NAME}.json"


# ---------------------------------------------------------------- using the frame
class Frame:
    """A fitted amb3r_floor frame: AMB3R native points and poses -> the frame."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.M = np.asarray(doc["native_to_frame"], dtype=np.float64)   # includes the scale
        self.s = float(doc["scale"]["value"])
        self.R = self.M[:3, :3] / self.s                                   # rotation only

    @classmethod
    def load(cls, path: Path | None = None) -> "Frame":
        path = path or frame_path()
        if not path.is_file():
            raise SystemExit(f"{path} missing; fit it first: python -m src.geometry.floor_frame --run <structure run>")
        return cls(json.loads(path.read_text()))

    def points(self, p_native) -> np.ndarray:
        p = np.asarray(p_native, dtype=np.float64)
        return p @ self.M[:3, :3].T + self.M[:3, 3]

    def pose(self, camera_to_world_native) -> np.ndarray:
        """A camera-to-world 4x4 (OpenCV axes): rotation turned, not scaled; centre scaled and moved."""
        T = np.asarray(camera_to_world_native, dtype=np.float64).reshape(4, 4)
        out = np.eye(4)
        out[:3, :3] = self.R @ T[:3, :3]
        out[:3, 3] = self.points(T[:3, 3])
        return out

    def block(self) -> dict:
        """What a handoff writes under `frame`."""
        d = self.doc
        return {"definition": d["definition"], "file": d["file"], "native_to_frame": d["native_to_frame"],
                "scale": d["scale"], "first_frame": d["first_frame"]}


def scannet_scan_to_frame(frame: Frame, sim3: dict) -> np.ndarray:
    """Rigid 4x4 taking ScanNet scan-frame points (and poses, by left-multiplying) into the frame.

    Evaluation and the GT producer only. `sim3` is the `sim3_oracle_scale` record that maps
    AMB3R into the scan frame (p_scan = s R_s p_n + t_s); its scale must be the frame's.
    """
    s = float(sim3["alignment_scale"])
    if abs(s - frame.s) > 1e-9:
        raise ValueError(f"the frame was scaled by {frame.s}, the Sim(3) by {s}: the scan frame does not "
                         "map rigidly into this frame")
    Rs = np.asarray(sim3["alignment_rotation"], dtype=np.float64)
    ts = np.asarray(sim3["alignment_translation_m"], dtype=np.float64)
    # p_f = R (s p_n) + M_t, and s p_n = Rs^T (p_scan - ts)
    S = np.eye(4)
    S[:3, :3] = frame.R @ Rs.T
    S[:3, 3] = frame.M[:3, 3] - frame.R @ Rs.T @ ts
    return S


# ---------------------------------------------------------------- fitting it
def _plane(p: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = p[rng.choice(len(p), min(len(p), SUBSAMPLE), replace=False)]
    best = None
    for _ in range(ITERS):
        s = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); length = np.linalg.norm(n)
        if length < 1e-9:
            continue
        n /= length
        inl = np.abs((sub - s[0]) @ n) < TOL
        if best is None or inl.sum() > best.sum():
            best = inl
    q = sub[best]
    c = q.mean(0)
    n = np.linalg.eigh(np.cov((q - c).T))[1][:, 0]
    return c, n, sub


def fit(floor_points: np.ndarray, poses: np.ndarray, scale: float, *, first: int = 0) -> dict:
    """The frame from floor points and camera poses, both in AMB3R's native frame.

    floor_points (N, 3) native; poses (F, 4, 4) native camera-to-world, OpenCV axes, in sequence
    order; `first` is the index the +X direction is taken from.
    """
    rng = np.random.default_rng(SEED)
    P = np.asarray(floor_points, dtype=np.float64) * scale
    c, n, _ = _plane(P, rng)
    centres = poses[:, :3, 3] * scale
    h = (centres - c) @ n
    if np.median(h) < 0:                               # up is the side the cameras are on
        n, h = -n, -h
    used = first
    while used < len(poses):
        f = poses[used, :3, :3] @ np.array([0.0, 0.0, 1.0])
        if abs(f @ n) <= MAX_PITCH:
            break
        used += 1
    else:
        raise ValueError("every camera looks straight up or down; no horizontal direction for +X")
    x = f - (f @ n) * n; x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n])
    c0 = centres[used]
    o = c0 - ((c0 - c) @ n) * n
    M = np.eye(4); M[:3, :3] = scale * R; M[:3, 3] = -R @ o
    res = np.abs((P - c) @ n)
    return {"native_to_frame": M, "rotation": R, "origin_scaled_native": o, "first_used": used,
            "first_pitch_deg": float(np.degrees(np.arcsin(np.clip(f @ n, -1, 1)))),
            "first_height_m": float((c0 - c) @ n),
            "floor_fit": {"points": int(len(P)), "tol_m": TOL, "subsample": SUBSAMPLE, "iters": ITERS, "seed": SEED,
                          "residual_median_cm": float(np.median(res) * 100),
                          "residual_p90_cm": float(np.percentile(res, 90) * 100),
                          "within_3cm": float(np.mean(res < 0.03))},
            "cameras_above_floor_m": {"count": int(len(h)), "median": float(np.median(h)), "min": float(h.min()),
                                      "below_floor": int((h < 0).sum())}}


def floor_points_of_run(run: str):
    """Floor-mask pixels of a structure segmentation run, as AMB3R native points.

    The same pixels `src.structure.lift` lifts: confidence >= CONF, finite, and with a
    finite grid normal (the normal stencil drops the grid border).
    """
    from PIL import Image

    from src.geometry.amb3r_mask_source import AMBR3R_INTERMEDIATE, AMBR3R_SIZE
    from src.geometry.backproject_masks import align_mask
    from src.structure.frames import CONF, PointMaps, frame_slots, masks_dir
    from src.structure.lift import grid_normals

    maps = PointMaps(frame=None)
    slot = frame_slots()
    out, frames, masks = [], set(), 0
    for folder in sorted(masks_dir(run).glob("frame_*")):
        result = json.loads((folder / "result.json").read_text())
        kept = [k for k in result["kept"] if k["label"] == "floor"]
        if not kept:
            continue
        fid = int(result["frame_id"]); seq = slot[fid]
        p = maps.native(seq)
        cam = maps.pose[seq][:3, 3].astype(np.float64)
        nrm = grid_normals(p.reshape(maps.H, maps.W, 3), cam).reshape(-1, 3)
        ok = (maps.conf[seq].reshape(-1) >= CONF) & np.isfinite(p).all(1) & np.isfinite(nrm).all(1)
        for k in kept:
            m = align_mask(np.asarray(Image.open(folder / k["mask_file"]).convert("L")) > 0,
                           AMBR3R_INTERMEDIATE, AMBR3R_SIZE).reshape(-1)
            out.append(p[m & ok]); masks += 1
        frames.add(fid)
    return np.concatenate(out), maps, {"masks": masks, "frames": len(frames)}


def oracle_scale(sim3_path: Path) -> dict:
    rec = json.loads(Path(sim3_path).read_text())["sim3_oracle_scale"]
    return {"value": float(rec["alignment_scale"]), "source": "oracle_gt_poses",
            "record": str(sim3_path), "field": "sim3_oracle_scale.alignment_scale",
            "note": "AMB3R's scale corrected by an Umeyama fit of its camera centres to ScanNet's GT camera "
                    "centres; only the scale is taken from GT, not the rotation or translation. A new scene "
                    "without GT poses has no such value"}


def build(run: str, scene_id: str = SCENE_ID, scale_source: str = "oracle_gt_poses",
          out: Path | None = None) -> dict:
    from src.geometry.amb3r_mask_source import DEFAULT_NPZ, DEFAULT_SIM3
    from src.structure.frames import frame_slots, masks_dir

    if scale_source not in SCALE_SOURCES:
        raise SystemExit(f"scale_source must be one of {SCALE_SOURCES}")
    scale = oracle_scale(DEFAULT_SIM3) if scale_source == "oracle_gt_poses" else {
        "value": 1.0, "source": "native", "note": "AMB3R metric head as it came out; not corrected"}
    P, maps, counts = floor_points_of_run(run)
    seq_to_fid = {v: k for k, v in frame_slots().items()}
    r = fit(P, maps.pose.astype(np.float64), scale["value"])
    out = out or frame_path(scene_id)
    doc = {
        "coordinate_frame": FRAME_NAME, "scene_id": scene_id, "definition": DEFINITION,
        "file": str(out.relative_to(config.ROOT)) if out.is_relative_to(config.ROOT) else str(out),
        "units": "meters", "up_axis": "Z", "handedness": "right",
        "native_to_frame": [[round(float(v), 12) for v in row] for row in r["native_to_frame"]],
        "native_to_frame_note": "p_frame = native_to_frame @ [p_native, 1] (includes the scale); a native "
                                "camera-to-world T maps to R @ T[:3,:3] and native_to_frame @ [T[:3,3], 1], "
                                "R = native_to_frame[:3,:3] / scale",
        "scale": scale,
        "first_frame": {"amb3r_index": r["first_used"], "source_frame_id": seq_to_fid.get(r["first_used"]),
                        "height_m": round(r["first_height_m"], 4), "pitch_deg": round(r["first_pitch_deg"], 2)},
        "floor_fit": {**r["floor_fit"], **counts, "segmentation": str(masks_dir(run)), "run": run},
        "cameras_above_floor_m": r["cameras_above_floor_m"],
        "inputs": {"point_maps": str(DEFAULT_NPZ), "conf_min": 0.3},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    f = r["floor_fit"]; cams = r["cameras_above_floor_m"]
    print(f"{out}: scale {scale['value']} ({scale['source']}); floor {f['points']} points from {counts['masks']} masks, "
          f"residual median {f['residual_median_cm']:.2f} cm; cameras {cams['median']:.2f} m above the floor "
          f"(min {cams['min']:.2f}); first frame {r['first_used']} at {r['first_height_m']:.2f} m, "
          f"pitch {r['first_pitch_deg']:.1f} deg")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="structure segmentation run whose floor masks define the floor")
    ap.add_argument("--scene", default=SCENE_ID)
    ap.add_argument("--scale-source", default="oracle_gt_poses", choices=SCALE_SOURCES)
    a = ap.parse_args()
    build(a.run, a.scene, a.scale_source)


if __name__ == "__main__":
    main()
