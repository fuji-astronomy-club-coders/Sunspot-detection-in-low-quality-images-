"""Tests for logarithmic brightness histograms."""

import unittest

import numpy as np

from src.package.qt_histogram import (
    calculate_brightness_histogram,
    log_scaled_counts,
)


class BrightnessHistogramTests(unittest.TestCase):
    """Verify histogram counts and logarithmic scaling."""

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


if __name__ == "__main__":
    unittest.main()
