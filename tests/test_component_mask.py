"""Tests for exact connected-component mask selection."""

import unittest

import numpy as np

from src.package.mask import (
    applyComponentMask,
    component_polygons,
    component_selection_details,
    polygon_coverage_mask,
)


class ConnectedComponentMaskTests(unittest.TestCase):
    """Verify label-based exclusion and reusable component outline geometry."""

    def setUp(self) -> None:
        self.labels = np.zeros((12, 14), dtype=np.int32)
        self.labels[2:5, 2:5] = 1
        self.labels[7, 8:11] = 2
        self.labels[8, 9] = 2

    def test_component_mask_excludes_only_selected_labels(self) -> None:
        count, binary, details = applyComponentMask(
            self.labels,
            {2},
            original_count=2,
            return_details=True,
        )

        self.assertEqual(count, 1)
        self.assertTrue(np.all(binary[self.labels == 2] == 255))
        self.assertTrue(np.all(binary[self.labels == 1] == 0))
        self.assertEqual(details["selection_method"], "connected_component")
        self.assertEqual(details["excluded_candidate_labels"], [2])
        self.assertEqual(details["regions"][0]["area_pixels"], 4)
        self.assertEqual(details["regions"][0]["excluded_candidate_labels"], [2])

    def test_unknown_and_background_labels_are_ignored(self) -> None:
        count, _binary, details = applyComponentMask(
            self.labels,
            {0, 999},
            original_count=2,
            return_details=True,
        )

        self.assertEqual(count, 2)
        self.assertEqual(details["excluded_candidate_labels"], [])
        self.assertEqual(details["regions"], [])

    def test_component_polygons_cover_selected_component_pixels(self) -> None:
        polygons = component_polygons(self.labels, {1, 2})
        coverage = polygon_coverage_mask(self.labels.shape, polygons)

        self.assertGreaterEqual(len(polygons), 2)
        self.assertTrue(np.all(coverage[self.labels > 0]))

    def test_component_details_include_geometry_and_polygon_data(self) -> None:
        details = component_selection_details(self.labels, [1])

        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["label"], 1)
        self.assertEqual(details[0]["area_pixels"], 9)
        self.assertAlmostEqual(details[0]["centroid_x_px"], 3.0)
        self.assertAlmostEqual(details[0]["centroid_y_px"], 3.0)
        self.assertTrue(details[0]["polygons"])


if __name__ == "__main__":
    unittest.main()
