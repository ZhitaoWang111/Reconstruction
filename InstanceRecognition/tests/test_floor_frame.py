"""The delivery frame amb3r_floor (src.geometry.floor_frame) on a synthetic scan, and on the real one."""
import json

import numpy as np
import pytest

from src.geometry import floor_frame as ff


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis); t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _scene(scale=1.25, Q=None, t=(0.4, -1.0, 2.0)):
    """A room in a known upright frame, seen in a 'native' frame that is turned, shifted and scaled.

    Native = (Q @ upright + t) / scale, so `fit` with that scale must recover the upright frame
    up to a turn about +z and a shift in the floor plane - which its own definition then fixes."""
    rng = np.random.default_rng(1)
    Q = _rot([0.3, -1.0, 0.5], 110) if Q is None else Q
    floor = np.c_[rng.uniform(-3, 3, 5000), rng.uniform(-3, 3, 5000), rng.normal(0, 0.005, 5000)]
    poses = []
    for k in range(20):                                   # cameras 1.3 m up, looking 20 deg down
        yaw = np.radians(15 * k)
        fwd = np.array([np.cos(yaw) * np.cos(np.radians(20)), np.sin(yaw) * np.cos(np.radians(20)), -np.sin(np.radians(20))])
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); down = np.cross(fwd, right)
        T = np.eye(4); T[:3, :3] = np.c_[right, down, fwd]; T[:3, 3] = [0.1 * k, -0.05 * k, 1.3]
        poses.append(T)
    to_native = lambda p: (p @ Q.T + t) / scale
    native_poses = []
    for T in poses:
        N = np.eye(4); N[:3, :3] = Q @ T[:3, :3]; N[:3, 3] = to_native(T[:3, 3]); native_poses.append(N)
    return to_native(floor), np.array(native_poses), poses


def test_fit_recovers_an_upright_frame_with_the_first_camera_on_x():
    floor, native_poses, poses = _scene()
    r = ff.fit(floor, native_poses, 1.25)
    F = ff.Frame({"native_to_frame": r["native_to_frame"].tolist(), "scale": {"value": 1.25}})
    up = F.points(floor)
    assert abs(np.median(up[:, 2])) < 0.003 and np.std(up[:, 2]) < 0.01           # the floor is z = 0
    T0 = F.pose(native_poses[0])
    assert np.allclose(T0[:2, 3], 0, atol=1e-3) and T0[2, 3] == pytest.approx(1.3, abs=0.01)   # origin under camera 0
    fwd = T0[:3, 2]
    assert fwd[1] == pytest.approx(0, abs=1e-3) and fwd[0] > 0                    # +X is camera 0's heading
    assert np.allclose(T0[:3, :3] @ T0[:3, :3].T, np.eye(3), atol=1e-9) and np.linalg.det(T0[:3, :3]) > 0
    # every camera keeps its height and its distance to the others (scale restored, nothing bent)
    C = np.array([F.pose(T)[:3, 3] for T in native_poses]); G = np.array([T[:3, 3] for T in poses])
    assert np.allclose(C[:, 2], 1.3, atol=0.01)
    assert np.allclose(np.linalg.norm(C - C[0], axis=1), np.linalg.norm(G - G[0], axis=1), atol=1e-3)


def test_up_is_the_side_the_cameras_are_on():
    """Flip the native frame upside down: +Z must still point from the floor to the cameras."""
    floor, native_poses, _ = _scene(Q=_rot([1, 0, 0], 180) @ _rot([0, 0, 1], 30))
    r = ff.fit(floor, native_poses, 1.25)
    assert r["cameras_above_floor_m"]["below_floor"] == 0 and r["cameras_above_floor_m"]["median"] > 1.2


def test_a_camera_looking_straight_down_is_skipped_for_x():
    Q = _rot([0.3, -1.0, 0.5], 110)
    floor, native_poses, _ = _scene(Q=Q)
    down_cam = np.c_[[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]]                      # optical axis = -z, upright frame
    native_poses[0, :3, :3] = Q @ down_cam
    assert ff.fit(floor, native_poses, 1.25)["first_used"] == 1


def test_scan_to_frame_is_rigid_and_agrees_with_the_points():
    floor, native_poses, _ = _scene()
    r = ff.fit(floor, native_poses, 1.25)
    F = ff.Frame({"native_to_frame": r["native_to_frame"].tolist(), "scale": {"value": 1.25}})
    Rs = _rot([0.2, 0.7, -0.1], 40); ts = np.array([3.0, -2.0, 0.5])
    sim3 = {"alignment_scale": 1.25, "alignment_rotation": Rs.tolist(), "alignment_translation_m": ts.tolist()}
    S = ff.scannet_scan_to_frame(F, sim3)
    p_scan = 1.25 * floor[:50] @ Rs.T + ts
    assert np.allclose(p_scan @ S[:3, :3].T + S[:3, 3], F.points(floor[:50]), atol=1e-9)
    with pytest.raises(ValueError, match="does not\nmap rigidly|does not map rigidly"):
        ff.scannet_scan_to_frame(F, {**sim3, "alignment_scale": 1.0})


real = pytest.mark.skipif(not ff.frame_path().is_file(), reason="amb3r_floor.json not fitted")


@real
def test_real_frame_file():
    doc = json.loads(ff.frame_path().read_text())
    assert doc["coordinate_frame"] == "amb3r_floor" and doc["scale"]["source"] == "oracle_gt_poses"
    assert doc["floor_fit"]["residual_median_cm"] < 2.0
    cams = doc["cameras_above_floor_m"]
    assert cams["below_floor"] == 0 and 1.0 < cams["median"] < 1.6
    F = ff.Frame(doc)
    assert np.allclose(F.R @ F.R.T, np.eye(3), atol=1e-9) and np.linalg.det(F.R) > 0
