"""Tests for the labeled sunspot bounding-box diagnostic PNG."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.package.utils import save_sunspot_bounding_box_png


class SunspotBoundingBoxPngTests(unittest.TestCase):
    """Check that the PNG uses the retained CSV labels and component bounds."""

    def test_exports_component_boxes_and_original_label_values(self) -> None:
        background = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[10:20, 5:15] = 3
        labels[30:42, 34:49] = 7

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "sunspot_bounding_boxes.png"
            count = save_sunspot_bounding_box_png(
                background,
                labels,
                output_path,
                metadata={"extraction_method": "test"},
            )

            self.assertEqual(count, 2)
            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as png:
                rendered = np.asarray(png.convert("RGB"))
                self.assertEqual(png.size, (64, 64))
                self.assertEqual(png.info["extraction_method"], "test")
                label_mapping = json.loads(png.info["label_mapping"])

            self.assertEqual(label_mapping["csv_file"], "sunspot_statistics.csv")
            self.assertEqual(label_mapping["csv_column"], "label")
            self.assertEqual(label_mapping["labels"], [3, 7])
            np.testing.assert_array_equal(rendered[10, 5], [0, 255, 0])
            np.testing.assert_array_equal(rendered[30, 34], [0, 255, 0])
            self.assertTrue(np.any(np.all(rendered == [255, 255, 0], axis=2)))

    def test_rejects_mismatched_image_sizes(self) -> None:
        background = np.zeros((10, 10), dtype=np.uint8)
        labels = np.zeros((9, 10), dtype=np.int32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "invalid.png"
            with self.assertRaises(ValueError):
                save_sunspot_bounding_box_png(background, labels, output_path)


if __name__ == "__main__":
    unittest.main()
