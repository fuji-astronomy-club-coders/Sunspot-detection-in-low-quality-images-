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

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_comparison(self, document: dict) -> None:
        """Write YAML and mirror its in-folder image references as test files."""
        for existing in self.date_folder.rglob("*"):
            if existing.is_file() and existing.suffix.lower() in comparison.IMAGE_SUFFIXES:
                existing.unlink()

        samples = document.get("samples", [])
        if isinstance(samples, list):
            for sample in samples:
                if not isinstance(sample, dict) or not isinstance(sample.get("image"), str):
                    continue
                image_path = Path(sample["image"])
                if image_path.is_absolute():
                    continue
                resolved = (self.date_folder / image_path).resolve()
                if (
                    resolved.is_relative_to(self.date_folder.resolve())
                    and resolved.suffix.lower() in comparison.IMAGE_SUFFIXES
                ):
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    resolved.write_bytes(b"placeholder")

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

    def test_additional_top_level_and_sample_keys_are_tolerated(self) -> None:
        self._write_comparison(
            {
                "date": "2026-11-02",
                "note": "extra",
                "samples": [
                    {
                        "source_id": "source_01",
                        "image": "source_01.tif",
                        "pixel_scale_arcsec_per_pixel": 0.6,
                        "exposure_s": 0.01,
                    },
                    {
                        "source_id": "source_04",
                        "image": "source_04.tif",
                        "pixel_scale_arcsec_per_pixel": 0.8,
                        "north_angle_deg": 90.0,
                        "flip_x": False,
                        "memo": "extra",
                    },
                ],
            }
        )

        with (
            patch.object(
                comparison.utils,
                "loadImage",
                side_effect=self._load_valid_image,
            ),
            self.assertLogs("sunspot.comparison", level="WARNING") as logs,
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        self.assertEqual([sample.error for sample in plan.samples], [None, None])
        log_text = "\n".join(logs.output)
        for ignored in ("date", "note", "exposure_s", "memo"):
            self.assertIn(ignored, log_text)

    def test_additional_keys_do_not_bypass_required_keys(self) -> None:
        self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_01",
                        "image": "source_01.tif",
                        "memo": "pixel scale is missing",
                    },
                    {
                        "source_id": "source_04",
                        "image": "source_04.tif",
                        "pixel_scale_arcsec_per_pixel": 0.8,
                        "north_angle_deg": 90.0,
                        "flip_x": False,
                    },
                ]
            }
        )

        with patch.object(
            comparison.utils,
            "loadImage",
            side_effect=self._load_valid_image,
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        self.assertIn("pixel_scale_arcsec_per_pixel", self._sample_error(plan, 0))
        self.assertIsNone(plan.samples[1].error)

    def test_comparison_accepts_one_entry_per_available_image(self) -> None:
        self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_01",
                        "image": "source_01.tif",
                        "pixel_scale_arcsec_per_pixel": 0.6,
                    },
                    {
                        "source_id": "source_04",
                        "image": "source_04.tif",
                        "pixel_scale_arcsec_per_pixel": 0.8,
                        "north_angle_deg": 90.0,
                        "flip_x": False,
                    },
                ]
            }
        )

        with patch.object(
            comparison.utils,
            "loadImage",
            side_effect=self._load_valid_image,
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        self.assertEqual(len(plan.samples), 2)
        self.assertTrue(all(sample.error is None for sample in plan.samples))

    def test_sample_count_must_match_supported_image_count(self) -> None:
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
        (self.date_folder / "extra_image.tif").write_bytes(b"placeholder")

        with self.assertRaisesRegex(comparison.ComparisonInputError, "画像数"):
            comparison.prepare_comparison(self.date_folder)

    def test_unregistered_source_id_is_a_comparison_error(self) -> None:
        self._write_comparison(
            {
                "samples": [
                    {
                        "source_id": "source_99",
                        "image": "source_99.tif",
                        "pixel_scale_arcsec_per_pixel": 0.6,
                    }
                ]
            }
        )

        with self.assertRaisesRegex(comparison.ComparisonInputError, "未登録の source_id"):
            comparison.prepare_comparison(self.date_folder)

    def test_compact_date_folder_name_is_rejected(self) -> None:
        compact_folder = self.source_root / "20261102"
        compact_folder.mkdir()

        with self.assertRaisesRegex(comparison.ComparisonInputError, "YYYY-MM-DD"):
            comparison.prepare_comparison(compact_folder)

    def test_duplicate_source_rows_are_all_skipped(self) -> None:
        first = {
            "source_id": "source_01",
            "image": "source_01.tif",
            "pixel_scale_arcsec_per_pixel": 0.6,
        }
        second = {
            **first,
            "image": "source_01_copy.tif",
        }
        self._write_comparison({"samples": [first, second]})

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
        (self.date_folder / "source_01.tif").write_bytes(b"placeholder")

        plan = comparison.prepare_comparison(self.date_folder)

        self.assertIn("外", self._sample_error(plan))


    def test_comparison_settings_read_configurable_units(self) -> None:
        """Tolerance is a solar-radius fraction; warning bounds are arcseconds."""
        settings = comparison._load_comparison_settings({
            "Comparison": {
                "matching": {"tolerance_normalized": 0.03},
                "angular_radius_warning_arcsec": {"min": 850.0, "max": 1050.0},
            }
        })

        self.assertEqual(settings["matching_tolerance_normalized"], 0.03)
        self.assertEqual(settings["angular_radius_warning_min_arcsec"], 850.0)
        self.assertEqual(settings["angular_radius_warning_max_arcsec"], 1050.0)

    def test_comparison_settings_reject_invalid_warning_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "min / max"):
            comparison._load_comparison_settings({
                "Comparison": {
                    "angular_radius_warning_arcsec": {"min": 1000.0, "max": 900.0}
                }
            })

    def test_recursive_sample_image_can_be_located_in_subfolder(self) -> None:
        nested_image = self.date_folder / "archive" / "camera_a.tif"
        nested_image.parent.mkdir(parents=True)
        nested_image.write_bytes(b"image placeholder")
        self._write_comparison({
            "samples": [{
                "source_id": "source_01",
                "image": "archive/camera_a.tif",
                "pixel_scale_arcsec_per_pixel": 0.6,
            }]
        })

        with patch.object(
            comparison.utils,
            "loadImage",
            return_value=np.zeros((32, 32), dtype=np.uint8),
        ):
            plan = comparison.prepare_comparison(self.date_folder)

        self.assertEqual(len(plan.samples), 1)
        self.assertIsNone(plan.samples[0].error)
        self.assertEqual(plan.samples[0].image_path, nested_image)

if __name__ == "__main__":
    unittest.main()
