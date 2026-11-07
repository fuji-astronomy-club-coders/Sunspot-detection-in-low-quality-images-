"""Regression tests for cached MIN2 result sidecars in the application."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.package import limb_darking_resolution as ldr


class Min2SavedResultTests(unittest.TestCase):
    """Verify optional reuse and validation of per-image MIN2 JSON results."""

    def setUp(self) -> None:
        self.image = np.zeros((32, 48), dtype=np.uint16)

    def _write_result(self, image_path: Path, *, shape: list[int] | None = None) -> Path:
        result_path = image_path.with_name(f"{image_path.stem}_min2.json")
        payload = {
            "format": "solar_disk_detection.min2_result",
            "format_version": 1,
            "source_image": {
                "name": image_path.name,
                "shape": shape if shape is not None else [32, 48],
            },
            "solar_geometry": {
                "center_x_px": 23.25,
                "center_y_px": 15.5,
                "radius_px": 14.75,
            },
            "min2_version": "MIN2 v2.5.1",
        }
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return result_path

    def test_matching_result_is_used_without_running_min2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sun.tiff"
            image_path.write_bytes(b"source")
            self._write_result(image_path)
            with patch("src.package.limb_darking_resolution.min2") as min2_mock:
                geometry = ldr.find_solar_geometry(
                    self.image,
                    min2_config={"load_saved_result": True},
                    image_path=image_path,
                )
            self.assertEqual(geometry, ((23.25, 15.5), 14.75))
            min2_mock.assert_not_called()

    def test_size_mismatch_falls_back_to_min2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sun.tiff"
            image_path.write_bytes(b"source")
            self._write_result(image_path, shape=[16, 24])
            with patch(
                "src.package.limb_darking_resolution.min2",
                return_value=((22.0, 15.0), 13.0),
            ) as min2_mock:
                geometry = ldr.find_solar_geometry(
                    self.image,
                    min2_config={"load_saved_result": True},
                    image_path=image_path,
                )
            self.assertEqual(geometry, ((22.0, 15.0), 13.0))
            min2_mock.assert_called_once()

    def test_setting_is_optional_and_defaults_to_computation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sun.tiff"
            image_path.write_bytes(b"source")
            self._write_result(image_path)
            with patch(
                "src.package.limb_darking_resolution.min2",
                return_value=((21.0, 16.0), 12.0),
            ) as min2_mock:
                geometry = ldr.find_solar_geometry(
                    self.image,
                    min2_config={},
                    image_path=image_path,
                )
            self.assertEqual(geometry, ((21.0, 16.0), 12.0))
            min2_mock.assert_called_once()

    def test_cache_option_requires_a_boolean(self) -> None:
        with patch("src.package.limb_darking_resolution.min2") as min2_mock:
            with self.assertRaises(ValueError):
                ldr.find_solar_geometry(
                    self.image,
                    min2_config={"load_saved_result": "true"},
                    image_path="sun.tiff",
                )
            min2_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
