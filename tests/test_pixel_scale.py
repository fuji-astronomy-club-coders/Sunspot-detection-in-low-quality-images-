"""Regression tests for diameter-relative image-processing parameters."""

import unittest

from src.package.pixel_scale import (
    resolve_pixel_area,
    resolve_pixel_length,
    resolve_pixel_parameters,
)


class PixelScaleTests(unittest.TestCase):
    """Verify that fractions preserve baseline values and scale with solar diameter."""

    def test_linear_fraction_uses_solar_diameter(self) -> None:
        self.assertEqual(resolve_pixel_length("3/1000", 1000, "kernel"), 3)
        self.assertEqual(resolve_pixel_length("3/1000", 2000, "kernel"), 6)

    def test_odd_kernel_rounding(self) -> None:
        self.assertEqual(resolve_pixel_length("3/1000", 2000, "kernel", odd=True), 5)
        self.assertEqual(resolve_pixel_length("15/1000", 2000, "kernel", odd=True), 29)

    def test_integer_remains_a_fixed_pixel_value(self) -> None:
        self.assertEqual(resolve_pixel_length(5, 2000, "distance"), 5)
        self.assertEqual(resolve_pixel_area(2, 2000, "area"), 2)

    def test_area_fraction_scales_with_diameter_squared(self) -> None:
        self.assertEqual(resolve_pixel_area("1/500000", 1000, "area"), 2)
        self.assertEqual(resolve_pixel_area("1/500000", 2000, "area"), 8)

    def test_application_defaults_resolve_to_previous_baseline(self) -> None:
        config = {
            "Emphasis": {
                "SobelFilter": {"kernelSize": "3/1000"},
                "BlackTopHat": {"kernelSize": "3/200"},
                "GaussianFilter": {"sigmaSize": "1/50"},
            },
            "Extraction": {
                "countComponents": {"min_area": "1/500000"},
                "watershed": {"min_distance": "1/200"},
            },
        }

        resolved, effective = resolve_pixel_parameters(config, 1000)

        self.assertEqual(resolved["Emphasis"]["SobelFilter"]["kernelSize"], 3)
        self.assertEqual(resolved["Emphasis"]["BlackTopHat"]["kernelSize"], 15)
        self.assertEqual(resolved["Emphasis"]["GaussianFilter"]["sigmaSize"], 20)
        self.assertEqual(resolved["Extraction"]["countComponents"]["min_area"], 2)
        self.assertEqual(resolved["Extraction"]["watershed"]["min_distance"], 5)
        self.assertEqual(effective["solar_diameter_px"], 1000.0)

    def test_invalid_values_are_rejected(self) -> None:
        for value in ("1", "1/0", "-1/2", "not-a-fraction"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_pixel_length(value, 1000, "kernel")


if __name__ == "__main__":
    unittest.main()
