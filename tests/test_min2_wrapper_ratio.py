"""Tests for forwarding ratio-based limb width through the application wrapper."""

import unittest

from src.package.limb_darking_resolution import _resolve_min2_kwargs


class Min2LimbWidthWrapperTests(unittest.TestCase):
    """Validate the public config spelling and legacy MIN2 argument mapping."""

    def test_ratio_is_forwarded_to_legacy_parameter(self) -> None:
        kwargs = _resolve_min2_kwargs({"limb_width": "1/20"})
        self.assertEqual(kwargs["limb_wigth"], "1/20")

    def test_integer_pixel_width_remains_supported(self) -> None:
        kwargs = _resolve_min2_kwargs({"limb_width": 24})
        self.assertEqual(kwargs["limb_wigth"], 24)

    def test_invalid_ratios_are_rejected(self) -> None:
        for value in ("1", "1/0", "0/0", "-1/2", "1/-2", "abc/2", "1/2/3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _resolve_min2_kwargs({"limb_width": value})


if __name__ == "__main__":
    unittest.main()
