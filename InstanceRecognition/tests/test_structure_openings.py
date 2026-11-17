"""Opening edges (src.structure.walls.opening_bounds) on hand-made ray grids."""
import numpy as np

from src.structure import walls as wl

R = wl.MIN_RAYS


def grid(n_along=12, n_z=20):
    return np.zeros((n_along, n_z), int), np.zeros((n_along, n_z), bool)


def test_doorway_reaches_the_floor_when_nothing_solid_is_under_it():
    solid, free = grid()
    solid[:, :] = R                     # wall everywhere ...
    solid[4:8, :17] = 0                 # ... except the doorway, whose lowest row nobody saw
    free[4:8, 1:17] = True
    b = wl.opening_bounds(solid, free, 4, 8, 1, 17)
    assert b == {"left": "solid", "right": "solid", "top": "solid", "bottom": "floor"}


def test_window_has_a_solid_sill():
    solid, free = grid()
    solid[:, :] = R
    solid[4:8, 8:15] = 0
    free[4:8, 8:15] = True
    assert wl.opening_bounds(solid, free, 4, 8, 8, 15)["bottom"] == "solid"


def test_a_side_at_the_end_of_the_wall_is_not_wall():
    solid, free = grid()
    solid[:, :] = R
    solid[0:4, :17] = 0
    free[0:4, 1:17] = True
    assert wl.opening_bounds(solid, free, 0, 4, 1, 17)["left"] == "extent_end"


def test_a_side_nobody_saw_is_unseen():
    solid, free = grid()
    solid[:, :] = R
    solid[3:8, :17] = 0                 # column 3 left unseen beside the opening
    free[4:8, 1:17] = True
    assert wl.opening_bounds(solid, free, 4, 8, 1, 17)["left"] == "unseen"
