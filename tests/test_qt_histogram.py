"""Tests for logarithmic brightness histograms and cursor interaction."""

import unittest

import numpy as np

from src.package.qt_histogram import (
    calculate_brightness_histogram,
    histogram_bin_index,
    log_scaled_counts,
)


class BrightnessHistogramTests(unittest.TestCase):
    """Verify histogram counts, logarithmic scaling, and cursor lookup."""

    def test_histogram_counts_only_finite_brightness_values(self) -> None:
        image = np.array(
            [[0.0, 1.0, 1.0], [10.0, np.nan, np.inf]],
            dtype=np.float64,
        )

        counts, _edges, minimum, maximum, pixel_count = (
            calculate_brightness_histogram(image)
        )

        self.assertEqual(int(counts.sum()), 4)
        self.assertEqual(pixel_count, 4)
        self.assertEqual(minimum, 0.0)
        self.assertEqual(maximum, 10.0)

    def test_constant_image_is_kept_in_a_single_histogram_bin(self) -> None:
        image = np.full((8, 8), 42.5, dtype=np.float32)

        counts, _edges, minimum, maximum, pixel_count = (
            calculate_brightness_histogram(image)
        )

        self.assertEqual(int(counts.max()), 64)
        self.assertEqual(int(np.count_nonzero(counts)), 1)
        self.assertEqual(pixel_count, 64)
        self.assertEqual(minimum, 42.5)
        self.assertEqual(maximum, 42.5)

    def test_log_scale_keeps_low_frequency_bins_visible(self) -> None:
        heights = log_scaled_counts(np.array([0, 1, 10, 10_000]))

        self.assertEqual(float(heights[0]), 0.0)
        self.assertGreater(float(heights[1]), 0.0)
        self.assertGreater(float(heights[2]), float(heights[1]))
        self.assertEqual(float(heights[3]), 1.0)

    def test_cursor_position_selects_the_corresponding_bin(self) -> None:
        self.assertEqual(histogram_bin_index(25.0, 0.0, 100.0, 4), 1)
        self.assertEqual(histogram_bin_index(99.9, 0.0, 100.0, 4), 3)
        self.assertEqual(histogram_bin_index(100.0, 0.0, 100.0, 4), 3)

    def test_cursor_outside_plot_does_not_select_a_bin(self) -> None:
        self.assertIsNone(histogram_bin_index(-0.1, 0.0, 100.0, 4))
        self.assertIsNone(histogram_bin_index(100.1, 0.0, 100.0, 4))

    def test_invalid_plot_geometry_does_not_select_a_bin(self) -> None:
        self.assertIsNone(histogram_bin_index(5.0, 0.0, 0.0, 4))
        self.assertIsNone(histogram_bin_index(5.0, 0.0, 10.0, 0))


if __name__ == "__main__":
    unittest.main()
