"""Tests for launcher argument selection."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import launcher_gui
from launcher_gui import build_batch_arguments


class LauncherArgumentTests(unittest.TestCase):
    """Verify that GUI selections map to supported run.bat arguments."""

    def test_single_image_mode_uses_no_run_py_arguments(self) -> None:
        self.assertEqual(build_batch_arguments("app"), [])

    def test_force_venv_option_is_forwarded(self) -> None:
        self.assertEqual(
            build_batch_arguments("app", use_venv=True),
            ["--venv"],
        )

    def test_comparison_mode_passes_the_selected_folder(self) -> None:
        self.assertEqual(
            build_batch_arguments(
                "app",
                compare=True,
                date_folder=r"E:\Solar Data\2026-11-02",
            ),
            ["--compare", r"E:\Solar Data\2026-11-02"],
        )

    def test_comparison_mode_without_folder_uses_run_py_dialog(self) -> None:
        self.assertEqual(
            build_batch_arguments("app", compare=True),
            ["--compare"],
        )

    def test_venv_and_comparison_options_can_be_combined(self) -> None:
        self.assertEqual(
            build_batch_arguments(
                "app",
                use_venv=True,
                compare=True,
                date_folder="source_input/2026-11-02",
            ),
            ["--venv", "--compare", "source_input/2026-11-02"],
        )

    def test_setup_venv_option(self) -> None:
        self.assertEqual(
            build_batch_arguments("setup_venv"),
            ["--setup"],
        )

    def test_setup_global_option(self) -> None:
        self.assertEqual(
            build_batch_arguments("setup_global"),
            ["--setup", "--global"],
        )

    def test_help_option(self) -> None:
        self.assertEqual(build_batch_arguments("help"), ["--help"])

    def test_unknown_action_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_batch_arguments("unknown")


class LauncherBatchInvocationTests(unittest.TestCase):
    """Verify Windows command-shell invocation of the selected batch command."""

    def test_batch_is_called_with_separate_quoted_arguments(self) -> None:
        batch_path = Path(r"C:\Projects With Spaces\run.bat")
        project_root = Path(r"C:\Projects With Spaces")
        selected_folder = r"E:\Solar Data & Archive\2026-11-02"

        with (
            patch.object(launcher_gui, "BATCH_FILE", batch_path),
            patch.object(launcher_gui, "PROJECT_ROOT", project_root),
            patch.object(
                launcher_gui.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=7),
            ) as run_command,
        ):
            exit_code = launcher_gui._run_batch(
                ["--compare", selected_folder]
            )

        self.assertEqual(exit_code, 7)
        expected_command = (
            f'call "{batch_path}" '
            f'"--_from-gui" "--compare" "{selected_folder}"'
        )
        run_command.assert_called_once_with(
            expected_command,
            shell=True,
            cwd=project_root,
            check=False,
        )


class BatchArgumentParserRegressionTests(unittest.TestCase):
    """Ensure the GUI-only marker is not forwarded as an application argument."""

    def test_non_venv_compare_does_not_use_shift_unsafe_percent_star(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        batch_source = (repository_root / "run.bat").read_text(encoding="utf-8")
        compare_branch = batch_source.split(
            'if /i "%~1"=="--compare" (', maxsplit=1
        )[1].split(")", maxsplit=1)[0]

        self.assertIn('set "APP_ARGS=--compare"', compare_branch)
        self.assertIn('set "APP_ARGS=--compare %2"', compare_branch)
        self.assertNotIn("%*", compare_branch)


if __name__ == "__main__":
    unittest.main()
