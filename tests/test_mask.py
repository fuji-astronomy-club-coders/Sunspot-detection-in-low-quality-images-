"""Tests for rectangular and radial mask boundary policies."""

import unittest

import numpy as np

from src.package import mask


class MaskBoundaryPolicyTests(unittest.TestCase):
    """Verify candidate handling at ordinary and radial mask boundaries."""

    def test_rectangular_overlap_policy_excludes_any_intersection(self) -> None:
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[5, 5:8] = 1

        count, _image, details = mask.applyMask(
            labels,
            [(7, 5, 1, 1)],
            1,
            return_details=True,
            boundary_policy="overlap",
        )

        self.assertEqual(count, 0)
        self.assertEqual(details["excluded_candidate_labels"], [1])

    def test_rectangular_centroid_policy_keeps_boundary_crossing_candidate(self) -> None:
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[5, 5:8] = 1

        count, _image, details = mask.applyMask(
            labels,
            [(7, 5, 1, 1)],
            1,
            return_details=True,
            boundary_policy="centroid",
        )

        self.assertEqual(count, 1)
        self.assertEqual(details["excluded_candidate_labels"], [])

    def test_radial_overlap_policy_excludes_if_any_pixel_is_outside(self) -> None:
        labels = np.zeros((21, 21), dtype=np.int32)
        labels[10, 17] = 1
        labels[10, 19] = 1

        count, _image, details = mask.applyRadialMask(
            labels,
            1,
            (10.0, 10.0),
            10.0,
            0.85,
            boundary_policy="overlap",
            return_details=True,
        )

        self.assertEqual(count, 0)
        self.assertEqual(details["excluded_candidate_labels"], [1])

    def test_radial_centroid_policy_uses_component_centroid(self) -> None:
        labels = np.zeros((21, 21), dtype=np.int32)
        labels[10, 17] = 1
        labels[10, 19] = 1

        count, _image, details = mask.applyRadialMask(
            labels,
            1,
            (10.0, 10.0),
            10.0,
            0.85,
            boundary_policy="centroid",
            return_details=True,
        )

        self.assertEqual(count, 1)
        self.assertEqual(details["excluded_candidate_labels"], [])

    def test_rectangular_contained_policy_excludes_only_fully_contained_spots(self) -> None:
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[5, 5:8] = 1
        labels[6, 7:10] = 2

        count, _image, details = mask.applyMask(
            labels,
            [(5, 5, 3, 1)],
            2,
            return_details=True,
            boundary_policy="contained",
        )

        self.assertEqual(count, 1)
        self.assertEqual(details["excluded_candidate_labels"], [1])

    def test_radial_contained_policy_excludes_only_fully_outer_spots(self) -> None:
        labels = np.zeros((25, 25), dtype=np.int32)
        labels[12, 20:22] = 1
        labels[12, 22:24] = 2

        count, _image, details = mask.applyRadialMask(
            labels,
            2,
            (12.0, 12.0),
            10.0,
            0.85,
            return_details=True,
            boundary_policy="contained",
        )

        self.assertEqual(count, 1)
        self.assertEqual(details["excluded_candidate_labels"], [2])

    def test_radial_ratio_above_one_is_allowed(self) -> None:
        labels = np.zeros((40, 40), dtype=np.int32)
        labels[15, 31] = 1

        count, _image, details = mask.applyRadialMask(
            labels,
            1,
            (15.0, 15.0),
            10.0,
            1.6,
            return_details=True,
        )

        self.assertEqual(count, 0)
        self.assertEqual(details["threshold_radius_px"], 16.0)

    def test_radial_ratio_must_be_positive_and_finite(self) -> None:
        labels = np.zeros((3, 3), dtype=np.int32)

        with self.assertRaises(ValueError):
            mask.applyRadialMask(labels, 0, (1.0, 1.0), 1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
