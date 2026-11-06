"""Tests for image-backed and transparent sunspot bounding-box outputs."""

import base64
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from src.package.utils import save_sunspot_bounding_box_files


class SunspotBoundingBoxOverlayTests(unittest.TestCase):
    """Check image backgrounds, transparent exports, SVG vectors, and labels."""

    def test_saves_image_background_and_vector_svg_with_csv_labels(self) -> None:
        reference = np.arange(96 * 96, dtype=np.uint16).reshape(96, 96)
        labels = np.zeros((96, 96), dtype=np.int32)
        labels[10:20, 5:15] = 3
        labels[30:42, 34:49] = 7

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "sunspot_bounding_boxes.png"
            count = save_sunspot_bounding_box_files(reference, labels, output_path)

            self.assertEqual(count, 2)
            svg_path = output_path.with_suffix(".svg")
            self.assertTrue(output_path.is_file())
            self.assertTrue(svg_path.is_file())

            with Image.open(output_path) as png:
                rgba = np.asarray(png.convert("RGBA"))
                self.assertEqual(png.size, (96, 96))
                label_mapping = json.loads(png.info["label_mapping"])
                layout = json.loads(png.info["label_layout"])

            self.assertEqual(label_mapping["csv_file"], "sunspot_statistics.csv")
            self.assertEqual(label_mapping["csv_column"], "label")
            self.assertEqual(label_mapping["labels"], [3, 7])
            self.assertTrue(np.all(rgba[:, :, 3] == 255))
            self.assertTrue(np.any(rgba[40:90, 40:90, :3] > 0))
            self.assertEqual(rgba[10, 8, 3], 255)
            self.assertGreater(int(rgba[10, 8, 1]), 200)
            self.assertGreater(int(rgba[10, 8, 1]), int(rgba[10, 8, 0]) * 2)
            self.assertEqual(len(layout), 2)

            root = ET.parse(svg_path).getroot()
            namespace = {"svg": "http://www.w3.org/2000/svg"}
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertEqual(len(root.findall(".//svg:rect", namespace)), 2)
            self.assertEqual(len(root.findall(".//svg:text", namespace)), 2)
            image_element = root.find(".//svg:image", namespace)
            self.assertIsNotNone(image_element)
            assert image_element is not None
            href = image_element.attrib["href"]
            self.assertTrue(href.startswith("data:image/png;base64,"))
            embedded_png = base64.b64decode(href.split(",", maxsplit=1)[1])
            self.assertTrue(embedded_png.startswith(b"\\x89PNG\\r\\n\\x1a\\n"))

    def test_can_export_transparent_background_when_requested(self) -> None:
        reference = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
        labels = np.zeros(reference.shape, dtype=np.int32)
        labels[10:14, 10:14] = 5

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "transparent.png"
            save_sunspot_bounding_box_files(
                reference,
                labels,
                output_path,
                options={"background_mode": "transparent", "save_svg": False},
            )
            with Image.open(output_path) as png:
                rgba = np.asarray(png.convert("RGBA"))

        self.assertEqual(rgba[0, 0, 3], 0)
        self.assertGreater(rgba[10, 10, 3], 0)

    def test_dense_labels_are_placed_without_overlap(self) -> None:
        reference = np.zeros((220, 260), dtype=np.uint16)
        labels = np.zeros(reference.shape, dtype=np.int32)
        positions = [
            (100, 112), (106, 112), (112, 112), (118, 112), (124, 112),
            (100, 118), (106, 118), (112, 118), (118, 118), (124, 118),
            (100, 124), (106, 124), (112, 124), (118, 124), (124, 124),
        ]
        for label_value, (x, y) in enumerate(positions, start=41):
            labels[y : y + 2, x : x + 2] = label_value

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "dense.png"
            save_sunspot_bounding_box_files(reference, labels, output_path)
            with Image.open(output_path) as png:
                layout = json.loads(png.info["label_layout"])

        self.assertEqual(len(layout), len(positions))
        rectangles = [
            (item["x"], item["y"], item["x"] + item["width"], item["y"] + item["height"])
            for item in layout
        ]
        for index, first in enumerate(rectangles):
            for second in rectangles[index + 1 :]:
                self.assertFalse(
                    first[0] < second[2]
                    and second[0] < first[2]
                    and first[1] < second[3]
                    and second[1] < first[3],
                    "Label layout rectangles must not overlap.",
                )

    def test_can_disable_png_or_svg_independently(self) -> None:
        reference = np.zeros((24, 24), dtype=np.uint8)
        labels = np.zeros(reference.shape, dtype=np.int32)
        labels[8:12, 8:12] = 5

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "overlay.png"
            count = save_sunspot_bounding_box_files(
                reference,
                labels,
                output_path,
                options={"save_png": False, "save_svg": True},
            )
            self.assertEqual(count, 1)
            self.assertFalse(output_path.exists())
            self.assertTrue(output_path.with_suffix(".svg").exists())

    def test_rejects_mismatched_image_sizes(self) -> None:
        reference = np.zeros((10, 10), dtype=np.uint8)
        labels = np.zeros((9, 10), dtype=np.int32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "invalid.png"
            with self.assertRaises(ValueError):
                save_sunspot_bounding_box_files(reference, labels, output_path)


if __name__ == "__main__":
    unittest.main()