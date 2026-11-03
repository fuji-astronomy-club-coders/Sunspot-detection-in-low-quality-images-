"""Tests for polygon masks used by comparison mode."""

import unittest

import numpy as np

from src.package.mask import applyPolygonMask, polygon_coverage_mask


class PolygonMaskTests(unittest.TestCase):
    """Verify polygon rasterization and candidate boundary policies."""

    def setUp(self) -> None:
        self.polygon = np.asarray(
            [[5.0, 4.0], [8.0, 4.0], [8.0, 7.0], [5.0, 7.0]],
            dtype=np.float64,
        )

    def test_overlap_excludes_candidate_with_one_pixel_inside(self) -> None:
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[5, 3:6] = 1

        count, _image, details = applyPolygonMask(
            labels,
            [self.polygon],
            1,
            return_details=True,
            boundary_policy="overlap",
        )

        self.assertEqual(count, 0)
        self.assertEqual(details["excluded_candidate_labels"], [1])

    def test_centroid_keeps_candidate_when_centroid_is_outside(self) -> None:
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[5, 3:6] = 1

        count, _image, details = applyPolygonMask(
            labels,
            [self.polygon],
            1,
            return_details=True,
            boundary_policy="centroid",
        )

        self.assertEqual(count, 1)
        self.assertEqual(details["excluded_candidate_labels"], [])

    def test_contained_uses_union_of_polygon_coverage(self) -> None:
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[5, 4] = 1
        labels[5, 6] = 1
        first = np.asarray([[3, 4], [5, 4], [5, 6], [3, 6]], dtype=np.float64)
        second = np.asarray([[5, 4], [7, 4], [7, 6], [5, 6]], dtype=np.float64)

        count, _image, details = applyPolygonMask(
            labels,
            [first, second],
            1,
            return_details=True,
            boundary_policy="contained",
        )

        self.assertEqual(count, 0)
        self.assertEqual(details["excluded_candidate_labels"], [1])

    def test_empty_polygon_list_leaves_candidates_unchanged(self) -> None:
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[3, 3] = 1

        count, _image, details = applyPolygonMask(
            labels,
            [],
            1,
            return_details=True,
        )

        self.assertEqual(count, 1)
        self.assertEqual(details["selected_region_count"], 0)

    def test_polygon_coverage_returns_boolean_image(self) -> None:
        coverage = polygon_coverage_mask((12, 12), [self.polygon])
        self.assertEqual(coverage.dtype, np.bool_)
        self.assertTrue(coverage[5, 5])
        self.assertFalse(coverage[1, 1])

    def test_half_pixel_edges_preserve_selected_roi_dimensions(self) -> None:
        polygon = np.asarray(
            [[4.5, 4.5], [7.5, 4.5], [7.5, 7.5], [4.5, 7.5]],
            dtype=np.float64,
        )
        coverage = polygon_coverage_mask((12, 12), [polygon])
        self.assertEqual(int(np.count_nonzero(coverage)), 9)
        self.assertTrue(np.all(coverage[5:8, 5:8]))
        self.assertFalse(coverage[4, 5])
        self.assertFalse(coverage[8, 5])


if __name__ == "__main__":
    unittest.main()
