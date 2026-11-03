"""Tests for watershed extraction image output."""

import unittest

import numpy as np

from src.package.watershed import watershedCount


class WatershedOutputTests(unittest.TestCase):
    """Ensure watershed returns binary pixels separately from component labels."""

    def test_returned_image_is_binary_and_labels_remain_separate(self) -> None:
        binary = np.full((48, 48), 255, dtype=np.uint8)
        binary[8:20, 8:20] = 0
        binary[28:40, 28:40] = 0

        count, result_image, labels = watershedCount(
            binary,
            min_area=1,
            min_distance=4,
        )

        self.assertGreaterEqual(count, 2)
        self.assertEqual(result_image.shape, binary.shape)
        self.assertEqual(result_image.dtype, np.dtype(np.uint8))
        self.assertTrue(set(np.unique(result_image)).issubset({0, 255}))
        self.assertGreater(int(labels.max()), 0)
        self.assertEqual(labels.shape, binary.shape)


if __name__ == "__main__":
    unittest.main()
