"""Wall geometry (src.structure): the ray test, the floor frame, and the closure on synthetic rooms."""
import numpy as np

from src.structure import close as cl
from src.structure import walls as wl
from src.structure.frames import Rays, rotation_to_z
from src.structure.lift import grid_normals


class _Rays(Rays):
    """Rays built in memory instead of read from scene_rays.npz."""

    def __init__(self, P, C):
        self.P, self.C, self.frames = np.asarray(P, float), np.asarray(C, float), 1


def _wall_rays(gap=None, cam=(0.0, -2.0, 1.3), x_range=(-2.0, 2.0)):
    """A wall along y = 0 seen from a camera at y = -2. Inside `gap` (x0, x1) the wall is missing,
    so those rays carry straight on, through the wall line, to a surface 3 m further back."""
    xs, zs = np.meshgrid(np.linspace(*x_range, 161), np.linspace(0.15, 2.6, 50))
    P = np.c_[xs.ravel(), np.zeros(xs.size), zs.ravel()]
    C = np.tile(cam, (len(P), 1))
    if gap is not None:
        hole = (P[:, 0] > gap[0]) & (P[:, 0] < gap[1])
        P[hole] = C[hole] + (P[hole] - C[hole]) * (5.0 / 2.0)    # same ray, ending 3 m behind the wall line
    return _Rays(P, C)


def test_ray_grid_marks_solid_and_see_through_cells():
    rays = _wall_rays(gap=(-0.5, 0.5))
    c, n = np.array([0.0, 0.0]), np.array([0.0, -1.0])     # facing the camera
    solid, through = wl.ray_grid(rays, c, n, -2.0, 2.0)
    cols = np.arange(solid.shape[0]) * wl.CELL - 2.0 + wl.CELL / 2
    in_gap = np.abs(cols) < 0.4
    assert (solid[in_gap] == 0).all() and (through[in_gap].sum(1) > 0).all()
    assert (solid[np.abs(cols) > 0.7].sum(1) > 0).all() and (through[np.abs(cols) > 0.7] == 0).all()


def test_ray_grid_ignores_rays_that_stop_short():
    """A surface in front of the wall (furniture) neither fills nor opens the wall."""
    rays = _wall_rays()
    rays.P[:, 1] = -0.8                                   # everything hits something 0.8 m in front
    solid, through = wl.ray_grid(rays, np.zeros(2), np.array([0.0, -1.0]), -2.0, 2.0)
    assert solid.sum() == 0 and through.sum() == 0


def test_rotation_to_z_keeps_heading():
    n = np.array([0.03, -0.02, 1.0]); n /= np.linalg.norm(n)
    R = rotation_to_z(n)
    assert np.allclose(R @ n, [0, 0, 1], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    x = R @ np.array([1.0, 0, 0])
    assert abs(np.degrees(np.arctan2(x[1], x[0]))) < 0.1   # a horizontal direction stays where it pointed


def test_thinning_does_not_depend_on_where_x_points():
    """A wall at 35 deg to +x thins to the same count once the cubes turn to the room's axis."""
    rng = np.random.default_rng(0)
    a, z = rng.uniform(0, 4, 40000), rng.uniform(0, 2.5, 40000)
    for deg in (0.0, 35.0):
        t = np.radians(deg)
        wall = np.c_[a * np.cos(t), a * np.sin(t), z]
        normals = np.tile([-np.sin(t), np.cos(t), 0.0], (len(wall), 1))
        axis = wl.room_axis(normals)
        assert abs(np.degrees(np.angle(np.exp(4j * (axis - t)))) / 4) < 1e-6
        n_turned = len(wl.voxel_thin(wall, axis=axis)[0])
        n_fixed = len(wl.voxel_thin(wall)[0])
        if deg == 0.0:
            ref = n_turned
        else:
            assert abs(n_turned / ref - 1) < 0.02 and n_fixed / ref > 1.08     # the fixed grid inflates it


def test_grid_normals_point_to_the_camera():
    ys, xs = np.mgrid[0:20, 0:20].astype(float)
    P = np.dstack([xs * 0.05, np.full_like(xs, 2.0), ys * 0.05])   # a plane y = 2
    n = grid_normals(P, np.array([0.5, 0.0, 0.5]), step=2)
    inner = n[2:-2, 2:-2].reshape(-1, 3)
    assert np.allclose(inner, [0, -1, 0], atol=1e-6)
    assert np.isnan(n[0, 0]).all()


def test_robust_vertical_trims_a_second_surface():
    rng = np.random.default_rng(0)
    wall = np.c_[rng.uniform(0, 3, 600), rng.normal(0, 0.005, 600), rng.uniform(0, 2.5, 600)]
    shelf = np.c_[rng.uniform(1, 1.6, 120), rng.normal(0.25, 0.005, 120), rng.uniform(0.3, 1.0, 120)]
    P = np.r_[wall, shelf]; N = np.tile([0, -1.0, 0], (len(P), 1))
    c, n, keep = wl.robust_vertical(P, N, np.arange(len(P)))
    assert abs(abs(n[1]) - 1) < 1e-3 and abs(c[1]) < 0.01
    assert (keep < 600).mean() > 0.99                    # the shelf points were trimmed


def test_seen_space_covers_the_ray_paths_only():
    P = np.array([[x, 3.0, 1.0] for x in np.linspace(-1, 1, 41)])
    C = np.tile([0.0, 0.0, 1.0], (len(P), 1))
    P = np.repeat(P, 4, 0); C = np.repeat(C, 4, 0)       # a cell counts as seen after SEEN_MIN ray samples
    seen, lo = cl.seen_space(P, C)
    def at(x, y):
        i, j = cl.cell_of(np.array([x, y]), lo, seen.shape)[0]
        return bool(seen[i, j])
    assert at(0.0, 1.5)            # between the camera and the wall
    assert not at(0.0, 3.5)        # behind the wall
    assert not at(2.0, 1.5)        # off to the side, no ray went there
