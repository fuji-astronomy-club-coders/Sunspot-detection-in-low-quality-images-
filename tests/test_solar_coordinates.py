"""Tests for common solar-disk coordinate transforms."""

import unittest

import numpy as np

from src.package.solar_coordinates import Orientation, SolarGeometry, SolarTransform


class SolarTransformTests(unittest.TestCase):
    """Verify cardinal directions, mirroring, and forward/inverse transforms."""

    def setUp(self) -> None:
        self.geometry = SolarGeometry(center=(100.0, 100.0), radius=50.0)

    def test_north_up_maps_image_right_to_positive_x(self) -> None:
        transform = SolarTransform(self.geometry, Orientation((0.0, 1.0)))
        self.assertEqual(transform.pixel_to_normalized(150.0, 100.0), (1.0, 0.0))
        self.assertEqual(transform.pixel_to_normalized(100.0, 50.0), (0.0, 1.0))

    def test_flip_x_reverses_only_horizontal_coordinate(self) -> None:
        transform = SolarTransform(
            self.geometry,
            Orientation((0.0, 1.0), flip_x=True),
        )
        self.assertEqual(transform.pixel_to_normalized(150.0, 100.0), (-1.0, 0.0))
        self.assertEqual(transform.pixel_to_normalized(100.0, 50.0), (0.0, 1.0))

    def test_north_right_rotates_the_common_basis(self) -> None:
        transform = SolarTransform(self.geometry, Orientation((1.0, 0.0)))
        self.assertEqual(transform.pixel_to_normalized(100.0, 150.0), (1.0, 0.0))
        self.assertEqual(transform.pixel_to_normalized(150.0, 100.0), (0.0, 1.0))

    def test_forward_and_inverse_transforms_are_consistent(self) -> None:
        transform = SolarTransform(
            SolarGeometry((213.5, 188.0), 97.0),
            Orientation((0.6, -0.8), flip_x=True),
        )
        point = (245.25, 144.75)
        normalized = transform.pixel_to_normalized(*point)
        restored = transform.normalized_to_pixel(*normalized)
        np.testing.assert_allclose(restored, point, atol=1e-10)

    def test_zero_north_vector_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Orientation((0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
