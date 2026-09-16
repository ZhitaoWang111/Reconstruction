import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backproject_masks import (
    align_mask,
    apply_sim3,
    crop_box,
    display_colors,
    membership_bits,
    read_point_ply,
    select_source_ply_pixels,
    write_ply,
)


PLY_VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("mask_bits", "<u4"), ("source_frame_id", "<u4"), ("pixel_index", "<u4"),
])


def read_binary_ply(path):
    data = path.read_bytes()
    marker = b"end_header\n"
    header_end = data.index(marker) + len(marker)
    header = data[:header_end].decode("ascii")
    count_line = next(line for line in header.splitlines() if line.startswith("element vertex "))
    count = int(count_line.rsplit(" ", 1)[1])
    vertex = np.frombuffer(data[header_end:], dtype=PLY_VERTEX_DTYPE, count=count)
    return header, vertex


class BackprojectMaskUnitTests(unittest.TestCase):
    def test_crop_box_matches_amb3r_center_crop_rounding(self):
        self.assertEqual(crop_box((640, 480), (518, 392)), (3, 0, 637, 480))

    def test_align_mask_resizes_to_intermediate_before_crop_and_pointmap_resize(self):
        original = np.zeros((968, 1296), dtype=np.uint8)
        original[:, 0:4] = 255
        original[:, -4:] = 255
        original[470:498, 634:662] = 255

        aligned = align_mask(original, intermediate_size=(640, 480), target_size=(518, 392))

        self.assertEqual(aligned.shape, (392, 518))
        self.assertFalse(aligned[:, 0].any())
        self.assertFalse(aligned[:, -1].any())
        self.assertTrue(aligned[190:202, 253:265].any())

    def test_membership_bits_preserve_overlapping_masks_without_winner(self):
        first = np.array([[True, True], [False, False]])
        second = np.array([[False, True], [True, False]])

        bits = membership_bits([first, second], (2, 2))

        np.testing.assert_array_equal(bits, np.array([[1, 3], [2, 0]], dtype=np.uint32))
        colors = display_colors(bits.reshape(-1), [
            {"color": [10, 20, 30]},
            {"color": [40, 50, 60]},
        ])
        np.testing.assert_array_equal(colors[1], np.array([235, 65, 210], dtype=np.uint8))

    def test_membership_bits_accept_exactly_thirty_two_masks(self):
        masks = [np.zeros((1, 1), dtype=bool) for _ in range(32)]
        masks[-1][0, 0] = True

        bits = membership_bits(masks, (1, 1))

        self.assertEqual(int(bits[0, 0]), 1 << 31)
        self.assertEqual(bits.dtype, np.uint32)

    def test_membership_bits_reject_more_than_thirty_two_masks(self):
        masks = [np.zeros((1, 1), dtype=bool) for _ in range(33)]

        with self.assertRaisesRegex(ValueError, "at most 32 masks"):
            membership_bits(masks, (1, 1))

    def test_same_class_instances_remain_separate_frame_local_bits(self):
        chair_a = np.array([[True, False, False]])
        chair_b = np.array([[False, True, False]])

        bits = membership_bits([chair_a, chair_b], (1, 3))

        np.testing.assert_array_equal(bits, np.array([[1, 2, 0]], dtype=np.uint32))

    def test_apply_sim3_transforms_finite_points_with_scale_rotation_translation(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float32)
        transform = {
            "alignment_scale": 2.0,
            "alignment_rotation": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            "alignment_translation_m": [10.0, 20.0, 30.0],
        }

        world = apply_sim3(points, transform)

        expected = np.array([[6.0, 22.0, 36.0], [9.0, 18.0, 38.0]], dtype=np.float32)
        self.assertTrue(np.isfinite(world).all())
        np.testing.assert_allclose(world, expected)

    def test_apply_sim3_rejects_nonfinite_transform(self):
        transform = {
            "alignment_scale": 1.0,
            "alignment_rotation": [[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]],
            "alignment_translation_m": [0.0, 0.0, 0.0],
        }

        with self.assertRaisesRegex(ValueError, "Nonfinite Sim"):
            apply_sim3(np.zeros((1, 3), dtype=np.float32), transform)

    def test_select_source_ply_pixels_replays_rng_choice_order(self):
        pointmap = np.arange(30, dtype=np.float32).reshape(2, 5, 3)
        confidence = np.array([[0.0, 0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.95, 0.2, 0.99]], dtype=np.float32)
        expected_rng = np.random.default_rng(123)
        expected = expected_rng.choice(np.array([1, 2, 3, 4, 7, 9]), 4, replace=False)

        selected = select_source_ply_pixels(
            pointmap,
            confidence,
            threshold=0.6,
            limit=4,
            rng=np.random.default_rng(123),
        )

        np.testing.assert_array_equal(selected, expected)

    def test_select_source_ply_pixels_filters_nonfinite_points_and_confidence(self):
        pointmap = np.zeros((1, 4, 3), dtype=np.float32)
        pointmap[0, 1, 0] = np.nan
        confidence = np.array([[0.7, 0.9, np.inf, 0.2]], dtype=np.float32)

        selected = select_source_ply_pixels(
            pointmap,
            confidence,
            threshold=0.5,
            limit=10,
            rng=np.random.default_rng(1),
        )

        np.testing.assert_array_equal(selected, np.array([0]))

    def test_write_ply_round_trips_binary_vertex_attributes(self):
        points = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
        colors = np.array([[255, 0, 7], [8, 9, 10]], dtype=np.uint8)
        bits = np.array([3, 1 << 31], dtype=np.uint32)
        frame_ids = np.array([162, 2048], dtype=np.uint32)
        pixel_indices = np.array([12, 345678], dtype=np.uint32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask_points.ply"
            write_ply(path, points, colors, bits, frame_ids, pixel_indices)
            header, vertex = read_binary_ply(path)

        self.assertIn("element vertex 2", header)
        self.assertIn("property uint mask_bits", header)
        np.testing.assert_allclose(np.column_stack([vertex["x"], vertex["y"], vertex["z"]]), points)
        np.testing.assert_array_equal(np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]), colors)
        np.testing.assert_array_equal(vertex["mask_bits"], bits)
        np.testing.assert_array_equal(vertex["source_frame_id"], frame_ids)
        np.testing.assert_array_equal(vertex["pixel_index"], pixel_indices)

    def test_read_point_ply_reads_points_and_colors_from_vertex_only_binary_ply(self):
        points = np.array([[0.25, 1.25, 2.25], [3.25, 4.25, 5.25]], dtype=np.float32)
        colors = np.array([[1, 2, 3], [250, 251, 252]], dtype=np.uint8)
        bits = np.array([1, 2], dtype=np.uint32)
        frame_ids = np.array([0, 11], dtype=np.uint32)
        pixel_indices = np.array([9, 99], dtype=np.uint32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "with_extra_properties.ply"
            write_ply(path, points, colors, bits, frame_ids, pixel_indices)
            loaded_points, loaded_colors = read_point_ply(path)

        self.assertEqual(loaded_points.dtype, np.float32)
        self.assertEqual(loaded_colors.dtype, np.uint8)
        np.testing.assert_allclose(loaded_points, points)
        np.testing.assert_array_equal(loaded_colors, colors)


if __name__ == "__main__":
    unittest.main()
