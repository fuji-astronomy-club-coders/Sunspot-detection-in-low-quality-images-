"""Tests for command-line entry-point behavior."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run


class ComparisonEntryPointLoggingTests(unittest.TestCase):
    """Verify that comparison startup records the selected input folder."""

    def test_selected_date_folder_is_logged_at_info_level(self) -> None:
        selected_folder = Path("/external/observations/2024-09-23")
        plan = SimpleNamespace(
            date_folder=selected_folder,
            comparison_date="2024-09-23",
            sources=(),
            samples=(),
            mask_template_path=None,
        )
        summary = {
            "overall_status": "success",
            "successful_sources": [],
            "samples": [],
        }
        logger = MagicMock()

        with tempfile.TemporaryDirectory() as temporary_directory:
            fake_run_logger = MagicMock()
            fake_run_logger.session_dir = Path(temporary_directory)
            with (
                patch.object(
                    sys,
                    "argv",
                    ["run.py", "--compare", str(selected_folder)],
                ),
                patch.object(run, "prepare_comparison", return_value=plan),
                patch.object(run, "RunLogger", return_value=fake_run_logger),
                patch.object(run, "run_comparison", return_value=summary),
                patch.object(run, "LOGGER", logger),
            ):
                exit_code = run.main()

        self.assertEqual(exit_code, 0)
        logger.info.assert_called_once_with(
            "Selected comparison date folder: %s",
            selected_folder,
        )


 
    def test_batch_folder_is_processed_and_summary_is_finalized(self) -> None:
        selected_folder = Path("/external/solar-images")
        image_files = [selected_folder / "image01.tiff"]
        summary = {
            "overall_status": "success",
            "total_image_count": 1,
            "successful_image_count": 1,
            "failed_image_count": 0,
            "summary_csv": "/output/batch_summary.csv",
            "images_directory": "/output/images",
            "images": [
                {
                    "status": "success",
                    "file_name": "image01.tiff",
                    "sunspot_count": 7,
                    "excluded_sunspot_count": 2,
                }
            ],
        }
        fake_logger = MagicMock()
        with (
            patch.object(sys, "argv", ["run.py", "--batch", str(selected_folder)]),
            patch.object(run, "discover_batch_images", return_value=image_files),
            patch.object(run, "RunLogger", return_value=fake_logger),
            patch.object(run, "run_batch_analysis", return_value=summary) as batch_runner,
        ):
            exit_code = run.main()

        self.assertEqual(exit_code, 0)
        batch_runner.assert_called_once_with(image_files, run.config, fake_logger)
        fake_logger.finish.assert_called_once_with(summary, status="success")

    def test_batch_folder_selection_cancel_returns_exit_code_two(self) -> None:
        with (
            patch.object(sys, "argv", ["run.py", "--batch"]),
            patch.object(run, "PROJECT_ROOT", Path("/project")),
            patch.object(run, "select_directory", return_value=None),
            patch.object(run, "RunLogger") as logger_class,
        ):
            exit_code = run.main()

        self.assertEqual(exit_code, 2)
        logger_class.assert_not_called()

    def test_batch_and_comparison_options_cannot_be_combined(self) -> None:
        with (
            patch.object(
                sys, "argv", ["run.py", "--batch", "/images", "--compare", "/date"]
            ),
            self.assertRaises(SystemExit) as error,
        ):
            run.main()
        self.assertEqual(error.exception.code, 2)

    def test_folder_selection_cancel_returns_exit_code_two(self) -> None:
        """Canceling the comparison folder chooser exits before creating a run."""
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(sys, "argv", ["run.py", "--compare"]),
            patch.object(run, "PROJECT_ROOT", Path(temporary_directory)),
            patch.object(run, "select_directory", return_value=None),
            patch.object(run, "RunLogger") as logger_class,
        ):
            exit_code = run.main()

        self.assertEqual(exit_code, 2)
        logger_class.assert_not_called()

if __name__ == "__main__":
    unittest.main()
