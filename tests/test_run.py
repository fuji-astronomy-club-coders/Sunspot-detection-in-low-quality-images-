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


    def test_folder_selection_cancel_returns_exit_code_two(self) -> None:
        """Canceling the comparison folder chooser exits before creating a run."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
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
