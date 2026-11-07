"""Tests for folder-based batch image processing."""

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.package.batch_analysis import (
    BatchInputError,
    discover_batch_images,
    run_batch_analysis,
)


class BatchImageDiscoveryTests(unittest.TestCase):
    """Verify extension filtering and deterministic, non-recursive scanning."""

    def test_discovers_supported_extensions_case_insensitively_in_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "b.FITS",
                "A.tiff",
                "c.fit",
                "d.fts",
                "e.tif",
                "ignore.png",
            ):
                (root / name).write_bytes(b"placeholder")
            (root / "nested").mkdir()
            (root / "nested" / "nested.tif").write_bytes(b"placeholder")

            self.assertEqual(
                [path.name for path in discover_batch_images(root)],
                ["A.tiff", "b.FITS", "c.fit", "d.fts", "e.tif"],
            )

    def test_missing_folder_and_empty_folder_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(BatchInputError):
                discover_batch_images(root / "missing")
            with self.assertRaises(BatchInputError):
                discover_batch_images(root)


class BatchRunTests(unittest.TestCase):
    """Verify cumulative outputs and continuation after an individual failure."""

    def test_summary_contains_counts_metadata_and_failed_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_files = []
            for name in ("01-good.tiff", "02-bad.fits"):
                path = root / name
                path.write_bytes(b"placeholder")
                image_files.append(path)

            session_dir = root / "session"
            session_dir.mkdir()
            logger = MagicMock()
            logger.session_dir = session_dir
            logger.data = {"events": []}

            def fake_process_image(
                *,
                config: dict[str, Any],
                run_logger: MagicMock,
                output_dir: Path,
            ) -> int:
                image_path = Path(config["Paths"]["input_image"])
                if image_path.name == "02-bad.fits":
                    raise RuntimeError("synthetic processing error")
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "sunspot_statistics.csv").write_text(
                    "sunspot_id,label\n1,1\n2,2\n",
                    encoding="utf-8",
                )
                (output_dir / "sunspot_exclusion_list.csv").write_text(
                    "sunspot_id,label\n1,4\n",
                    encoding="utf-8",
                )
                return 2

            original_config = {"Paths": {"input_image": "original.tiff"}}
            with patch(
                "src.package.batch_analysis.process_image",
                side_effect=fake_process_image,
            ):
                summary = run_batch_analysis(image_files, original_config, logger)

            self.assertEqual(summary["overall_status"], "partial_success")
            self.assertEqual(summary["successful_image_count"], 1)
            self.assertEqual(summary["failed_image_count"], 1)
            self.assertEqual(
                [row["status"] for row in summary["images"]],
                ["success", "failed"],
            )
            self.assertEqual(summary["images"][0]["sunspot_count"], 2)
            self.assertEqual(summary["images"][0]["excluded_sunspot_count"], 1)
            self.assertEqual(
                original_config["Paths"]["input_image"],
                "original.tiff",
            )

            summary_path = Path(summary["summary_csv"])
            with summary_path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["file_name"], "01-good.tiff")
            self.assertEqual(rows[0]["sunspot_count"], "2")
            self.assertEqual(rows[1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
