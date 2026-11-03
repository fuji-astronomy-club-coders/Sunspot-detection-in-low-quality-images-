"""Tests for final-preview exclusion overlays."""

import unittest

import numpy as np

from src.package.qt_preview import (
    _format_cursor_readout,
    build_invalid_sunspot_mask,
)


class FinalPreviewCursorReadoutTests(unittest.TestCase):
    """Verify that the preview HUD reports source-image pixel values."""

    def test_displays_original_unsigned_integer_brightness(self) -> None:
        image = np.array([[0, 4095], [1024, 65535]], dtype=np.uint16)

        readout = _format_cursor_readout(image, 1, 1)

        self.assertEqual(readout, "Cursor: x=1, y=1, brightness=65535")

    def test_displays_floating_point_brightness_without_colormap_conversion(self) -> None:
        image = np.array([[0.0, 1.234567]], dtype=np.float32)

        readout = _format_cursor_readout(image, 1, 0)

        self.assertEqual(
            readout,
            f"Cursor: x=1, y=0, brightness={image[0, 1]}",
        )


class InvalidSunspotPreviewMaskTests(unittest.TestCase):
    """Verify preview overlays include candidates excluded by every mask type."""

    def test_combines_rectangular_and_radial_exclusions(self) -> None:
        labels = np.array(
            [
                [0, 1, 1, 0],
                [2, 2, 0, 3],
                [0, 4, 4, 0],
            ],
            dtype=np.int32,
        )

        invalid_mask = build_invalid_sunspot_mask(
            labels,
            [1, 2],
            [2, 3],
        )

        expected = np.array(
            [
                [False, True, True, False],
                [True, True, False, True],
                [False, False, False, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(invalid_mask, expected)

    def test_rmask_only_exclusions_are_included(self) -> None:
        labels = np.array([[0, 1, 2], [3, 0, 4]], dtype=np.int32)

        invalid_mask = build_invalid_sunspot_mask(labels, [], [2, 4])

        expected = np.array(
            [[False, False, True], [False, False, True]],
            dtype=bool,
        )
        np.testing.assert_array_equal(invalid_mask, expected)

    def test_empty_exclusions_return_an_empty_mask(self) -> None:
        labels = np.array([[0, 1], [2, 0]], dtype=np.int32)

        invalid_mask = build_invalid_sunspot_mask(labels, [], [])

        self.assertEqual(invalid_mask.dtype, np.dtype(bool))
        self.assertFalse(np.any(invalid_mask))


if __name__ == "__main__":
    unittest.main()
