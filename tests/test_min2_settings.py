"""Tests for configurable MIN2 solar geometry estimation."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.package import limb_darking_resolution as ldr


class Min2ConfigurationTests(unittest.TestCase):
    """Verify MIN2 config validation and forwarding to the legacy function."""

    def setUp(self) -> None:
        """Prepare a small image for the MIN2 wrapper tests."""
        self.image = np.zeros((32, 32), dtype=np.uint8)

    @patch("src.package.limb_darking_resolution.min2")
    def test_configured_values_are_forwarded(self, min2_mock: MagicMock) -> None:
        """All public settings are passed to MIN2 using its legacy width name."""
        min2_mock.return_value = ((15.5, 16.5), 12.0)
        settings = {
            "n": 18,
            "light_threshold": 42,
            "limb_width": 30,
            "iter_cycles": 4,
            "show": True,
            "debug": True,
            "show_simple": True,
        }

        geometry = ldr.find_solar_geometry(self.image, min2_config=settings)

        self.assertEqual(geometry, ((15.5, 16.5), 12.0))
        min2_mock.assert_called_once_with(
            self.image,
            n=18,
            light_threshold=42,
            limb_wigth=30,
            iter_cycles=4,
            show=True,
            debug=True,
            show_simple=True,
        )

    @patch("src.package.limb_darking_resolution.min2")
    def test_missing_settings_keep_existing_defaults(self, min2_mock: MagicMock) -> None:
        """Omitting MIN2 config preserves the function's existing defaults."""
        min2_mock.return_value = ((15.0, 16.0), 12.0)

        ldr.find_solar_geometry(self.image)

        min2_mock.assert_called_once_with(
            self.image,
            n=10,
            light_threshold=50,
            limb_wigth="1/60",
            iter_cycles=2,
            show=False,
            debug=False,
            show_simple=False,
        )

    def test_invalid_integer_settings_are_rejected(self) -> None:
        """Integer settings reject booleans, non-integers, and out-of-range values."""
        invalid_settings = (
            {"n": 1},
            {"n": True},
            {"light_threshold": -1},
            {"light_threshold": 50.5},
            {"limb_width": -1},
            {"iter_cycles": -1},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings), patch(
                "src.package.limb_darking_resolution.min2"
            ) as min2_mock:
                with self.assertRaises(ValueError):
                    ldr.find_solar_geometry(self.image, min2_config=settings)
                min2_mock.assert_not_called()

    def test_invalid_boolean_and_unknown_settings_are_rejected(self) -> None:
        """Boolean settings are strict and unknown keys catch spelling mistakes."""
        invalid_settings = ({"show": 1}, {"debug": "true"}, {"limb_wigth": 24})
        for settings in invalid_settings:
            with self.subTest(settings=settings), patch(
                "src.package.limb_darking_resolution.min2"
            ) as min2_mock:
                with self.assertRaises(ValueError):
                    ldr.find_solar_geometry(self.image, min2_config=settings)
                min2_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
