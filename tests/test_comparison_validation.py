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
        self.source_root = self.root / "external_observatory_data"
        self.date_folder = self.source_root / "2026-11-02"
        self.date_folder.mkdir(parents=True)
        self.sources = [
            {
                "source_id": f"source_0{index}",
                "display_name": f"観測所{index}",
                "orientation_policy": "fixed",
                "north_vector": {"x": 0.0, "y": 1.0},
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
        self._root_patch.start()
        self.addCleanup(self._root_patch.stop)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_comparison(self, document: dict) -> None:
        (self.date_folder / "comparison.yaml").write_text(
            yaml.safe_dump(document, allow_unicode=True),
            encoding="utf-8",
        )

    def _load_valid_image(self, _path: Path) -> np.ndarray:
        return np.zeros((10, 10), dtype=np.uint16)

    def _sample_error(self, plan: comparison.ComparisonPlan, index: int = 0) -> str:
        """Return one prepared sample's validation error for assertions."""
        error = plan.samples[index].error
        self.assertIsNotNone(error)
        assert error is not None
        return error

    def test_valid_profile_produces_one_prepared_sample(self) -> None:
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
        with patch.object(
            comparison.utils,
            "loadImage",
            side_effect=self._load_valid_image,
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        self.assertEqual(plan.comparison_date, "2026-11-02")
        self.assertEqual(plan.source_input_root, self.source_root.resolve())
        self.assertEqual(len(plan.sources), 4)
        sample = plan.samples[0]
        self.assertEqual(sample.source_id, "source_01")
        self.assertIsNone(sample.error)
        assert sample.orientation is not None
        self.assertEqual(sample.orientation.north_vector, (0.0, 1.0))

    def test_per_sample_angle_360_is_equivalent_to_zero_degrees(self) -> None:
        self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_04",
                        "image": "source_04.tif",
                        "pixel_scale_arcsec_per_pixel": 0.8,
                        "north_angle_deg": 360.0,
                        "flip_x": False,
                    }
                ]
            }
        )
        with patch.object(
            comparison.utils,
            "loadImage",
            side_effect=self._load_valid_image,
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        sample = plan.samples[0]
        self.assertIsNone(sample.error)
        assert sample.orientation is not None
        np.testing.assert_allclose(sample.orientation.north_vector, (1.0, 0.0), atol=1e-12)
        self.assertAlmostEqual(sample.orientation.north_angle_deg, 0.0)

    def test_fixed_source_accepts_north_angle_deg(self) -> None:
        self.sources[0].pop("north_vector")
        self.sources[0]["north_angle_deg"] = 90.0
        (self.source_root / "sources.yaml").write_text(
            yaml.safe_dump({"sources": self.sources}, allow_unicode=True),
            encoding="utf-8",
        )
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
        with patch.object(
            comparison.utils,
            "loadImage",
            side_effect=self._load_valid_image,
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        sample = plan.samples[0]
        self.assertIsNone(sample.error)
        assert sample.orientation is not None
        np.testing.assert_allclose(sample.orientation.north_vector, (0.0, 1.0), atol=1e-12)
        self.assertAlmostEqual(sample.orientation.north_angle_deg, 90.0)

    def test_per_sample_angle_outside_zero_to_360_is_rejected(self) -> None:
        self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_04",
                        "image": "source_04.tif",
                        "pixel_scale_arcsec_per_pixel": 0.8,
                        "north_angle_deg": 361.0,
                        "flip_x": False,
                    }
                ]
            }
        )
        with patch.object(
            comparison.utils,
            "loadImage",
            side_effect=self._load_valid_image,
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        self.assertIn("0～360", self._sample_error(plan))
        self.assertIsNone(plan.samples[0].orientation)

    def test_orientation_rejects_both_vector_and_angle(self) -> None:
        self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_04",
                        "image": "source_04.tif",
                        "pixel_scale_arcsec_per_pixel": 0.8,
                        "north_vector": {"x": 0.0, "y": 1.0},
                        "north_angle_deg": 90.0,
                        "flip_x": False,
                    }
                ]
            }
        )
        with patch.object(
            comparison.utils,
            "loadImage",
            side_effect=self._load_valid_image,
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        self.assertIn("どちらか一方", self._sample_error(plan))
        self.assertIsNone(plan.samples[0].orientation)

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

        self.assertIn("読み込めません", self._sample_error(plan))

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

        with patch.object(
            comparison.utils,
            "loadImage",
            side_effect=self._load_valid_image,
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        self.assertIn("north_vector", self._sample_error(plan))
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

        error_texts = [self._sample_error(plan, index) for index in range(len(plan.samples))]
        self.assertTrue(all("重複" in error for error in error_texts))

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

        self.assertIn("外", self._sample_error(plan))


if __name__ == "__main__":
    unittest.main()
