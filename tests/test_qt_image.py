"""Tests for intensity-based Qt display image conversion."""

import unittest

import numpy as np

from src.package.qt_image import colorize_intensity


class IntensityDisplayTests(unittest.TestCase):
    """Verify colour maps are applied to pixel brightness rather than labels."""

    def test_colormap_preserves_image_shape_and_original_range(self) -> None:
        image = np.array([[0.0, 5.0], [10.0, 20.0]], dtype=np.float32)

        display, minimum, maximum = colorize_intensity(image, "HOT")

        self.assertEqual(display.shape, (2, 2, 3))
        self.assertEqual(display.dtype, np.dtype(np.uint8))
        self.assertEqual(minimum, 0.0)
        self.assertEqual(maximum, 20.0)
        self.assertFalse(np.array_equal(display[0, 0], display[1, 1]))

    def test_non_finite_values_do_not_change_the_intensity_range(self) -> None:
        image = np.array([[2.0, np.nan], [np.inf, 8.0]])

        _display, minimum, maximum = colorize_intensity(image, "VIRIDIS")

        self.assertEqual(minimum, 2.0)
        self.assertEqual(maximum, 8.0)

    def test_unknown_colormap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            colorize_intensity(np.ones((2, 2)), "NOT_A_COLORMAP")


if __name__ == "__main__":
    unittest.main()
