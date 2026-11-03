"""Tests for comparison-mode input validation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from src.package import comparison


class ComparisonValidationTests(unittest.TestCase):
    """Verify comparison profile and sample validation before run creation."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source_root = self.root / "source_input"
        self.date_folder = self.source_root / "2026-11-02"
        self.date_folder.mkdir(parents=True)
        self.sources = [
            {
                "source_id": f"source_0{index}",
                "display_name": f"観測所{index}",
                "orientation_policy": "fixed",
                "north_vector": {"x": 0.0, "y": -1.0},
                "flip_x": False,
            }
            for index in range(1, 4)
        ]
        self.sources.append(
            {
                "source_id": "source_04",
                "display_name": "観測所4",
                "orientation_policy": "per_sample",
            }
        )
        (self.source_root / "sources.yaml").write_text(
            yaml.safe_dump({"sources": self.sources}, allow_unicode=True),
            encoding="utf-8",
        )
        (self.date_folder / "source_01.tif").write_bytes(b"placeholder")
        (self.date_folder / "source_04.tif").write_bytes(b"placeholder")
        self._root_patch = patch.object(comparison, "PROJECT_ROOT", self.root)
        self._source_patch = patch.object(
            comparison,
            "SOURCE_INPUT_ROOT",
            self.source_root,
        )
        self._root_patch.start()
        self._source_patch.start()
        self.addCleanup(self._root_patch.stop)
        self.addCleanup(self._source_patch.stop)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_comparison(self, document: dict) -> None:
        (self.date_folder / "comparison.yaml").write_text(
            yaml.safe_dump(document, allow_unicode=True),
            encoding="utf-8",
        )

    def _load_valid_image(self, _path: Path) -> np.ndarray:
        return np.zeros((10, 10), dtype=np.uint16)

    def test_valid_profile_produces_one_prepared_sample(self) -> None:
        with patch.object(comparison.utils, "loadImage", side_effect=self._load_valid_image):
            self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_01",
                        "image": "source_01.tif",
                        "pixel_scale_arcsec_per_pixel": 0.6,
                    }
                ]
            }
        )

            plan = comparison.prepare_comparison(self.date_folder)

            self.assertEqual(plan.comparison_date, "2026-11-02")
            self.assertEqual(len(plan.sources), 4)
            self.assertEqual(plan.samples[0].source_id, "source_01")
            self.assertIsNone(plan.samples[0].error)
            self.assertEqual(plan.samples[0].orientation.north_vector, (0.0, -1.0))

    def test_unreadable_image_is_skipped_before_processing(self) -> None:
        self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_01",
                        "image": "source_01.tif",
                        "pixel_scale_arcsec_per_pixel": 0.6,
                    }
                ]
            }
        )
        with patch.object(comparison.utils, "loadImage", side_effect=ValueError("broken image")):
            plan = comparison.prepare_comparison(self.date_folder)

        self.assertIn("読み込めません", plan.samples[0].error)

    def test_per_sample_source_requires_orientation_fields(self) -> None:
        self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_04",
                        "image": "source_04.tif",
                        "pixel_scale_arcsec_per_pixel": 0.8,
                    }
                ]
            }
        )

        plan = comparison.prepare_comparison(self.date_folder)

        self.assertIn("north_vector", plan.samples[0].error)
        self.assertIsNone(plan.samples[0].orientation)

    def test_unknown_top_level_key_is_a_set_error(self) -> None:
        self._write_comparison(
            {
                "date": "2026-11-02",
                "samples": [
                    {
                        "source_id": "source_01",
                        "image": "source_01.tif",
                        "pixel_scale_arcsec_per_pixel": 0.6,
                    }
                ],
            }
        )

        with self.assertRaises(comparison.ComparisonInputError):
            comparison.prepare_comparison(self.date_folder)

    def test_duplicate_source_rows_are_all_skipped(self) -> None:
        sample = {
            "source_id": "source_01",
            "image": "source_01.tif",
            "pixel_scale_arcsec_per_pixel": 0.6,
        }
        self._write_comparison({"samples": [sample, sample]})

        plan = comparison.prepare_comparison(self.date_folder)

        self.assertTrue(all(row.error for row in plan.samples))
        self.assertTrue(all("重複" in row.error for row in plan.samples))

    def test_image_cannot_escape_date_folder(self) -> None:
        self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_01",
                        "image": "../outside.tif",
                        "pixel_scale_arcsec_per_pixel": 0.6,
                    }
                ]
            }
        )

        plan = comparison.prepare_comparison(self.date_folder)

        self.assertIn("外", plan.samples[0].error)


if __name__ == "__main__":
    unittest.main()
