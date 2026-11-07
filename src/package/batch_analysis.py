"""Batch processing for all supported solar images in one folder."""

from __future__ import annotations

import copy
import csv
import logging
import re
from pathlib import Path
from typing import Any

from src.main import process_image
from src.package.run_logger import RunLogger

LOGGER = logging.getLogger("sunspot.batch")
SUPPORTED_IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".fits", ".fit", ".fts"})
SUMMARY_CSV_NAME = "batch_summary.csv"
SUMMARY_FIELDS = [
    "file_name",
    "file_path",
    "file_type",
    "status",
    "image_width_px",
    "image_height_px",
    "image_dtype",
    "sunspot_count",
    "excluded_sunspot_count",
    "statistics_csv",
    "exclusion_list_csv",
    "output_directory",
    "error",
]


class BatchInputError(ValueError):
    """Raised when a folder does not contain supported image files."""


def discover_batch_images(folder: str | Path) -> list[Path]:
    """Return supported image files directly inside a selected folder.

    The scan is non-recursive and the result is sorted by case-insensitive
    filename, with the original filename used as a deterministic tie-breaker.
    """
    input_folder = Path(folder).expanduser().resolve()
    if not input_folder.is_dir():
        raise BatchInputError(f"選択したフォルダーが存在しません: {input_folder}")

    image_files = sorted(
        (
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not image_files:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise BatchInputError(
            f"フォルダー内に対応画像がありません: {input_folder} "
            f"(対応拡張子: {supported})"
        )
    return image_files


def _count_csv_rows(path: Path) -> int:
    """Count data rows in one UTF-8-BOM CSV output."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def _image_dimensions(
    events: list[dict[str, Any]],
) -> tuple[int | None, int | None, str | None]:
    """Read image dimensions and dtype from the latest metadata event."""
    for event in reversed(events):
        if event.get("event") != "input_image_metadata":
            continue
        details = event.get("details", {})
        array_info = details.get("array", {}) if isinstance(details, dict) else {}
        width = array_info.get("width_px")
        height = array_info.get("height_px")
        dtype = array_info.get("dtype")
        return (
            int(width) if width is not None else None,
            int(height) if height is not None else None,
            str(dtype) if dtype is not None else None,
        )
    return None, None, None


def _safe_output_stem(stem: str) -> str:
    """Make a filename stem safe as a Windows and POSIX directory name."""
    value = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
    return value or "image"


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the cumulative batch summary after each input image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_batch_analysis(
    image_files: list[Path],
    config: dict[str, Any],
    run_logger: RunLogger,
) -> dict[str, Any]:
    """Process every image independently and create one cumulative summary CSV.

    A failure for one image is recorded and does not prevent later images from
    being processed. Each image receives its own output subdirectory while all
    events remain in the same run log/session.
    """
    if not image_files:
        raise BatchInputError("一括解析する画像ファイルがありません。")

    image_output_root = run_logger.session_dir / "images"
    image_output_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_logger.session_dir / SUMMARY_CSV_NAME
    rows: list[dict[str, Any]] = []

    run_logger.record_event(
        "batch_analysis_started",
        {
            "input_folder": str(image_files[0].parent.resolve()),
            "image_count": len(image_files),
            "supported_extensions": sorted(SUPPORTED_IMAGE_SUFFIXES),
            "recursive": False,
            "summary_csv": str(summary_path),
        },
    )

    for index, image_path in enumerate(image_files, start=1):
        image_path = image_path.resolve()
        output_dir = image_output_root / (
            f"{index:04d}_{_safe_output_stem(image_path.stem)}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        image_config = copy.deepcopy(config)
        image_config.setdefault("Paths", {})["input_image"] = str(image_path)
        event_start = len(run_logger.data.get("events", []))

        LOGGER.info(
            "一括解析 [%d/%d]: %s",
            index,
            len(image_files),
            image_path.name,
        )
        try:
            sunspot_count = int(
                process_image(
                    config=image_config,
                    run_logger=run_logger,
                    output_dir=output_dir,
                )
            )
            statistics_path = output_dir / "sunspot_statistics.csv"
            exclusion_path = output_dir / "sunspot_exclusion_list.csv"
            if not statistics_path.is_file() or not exclusion_path.is_file():
                missing_path = (
                    statistics_path
                    if not statistics_path.is_file()
                    else exclusion_path
                )
                raise FileNotFoundError(f"解析CSVが出力されていません: {missing_path}")

            excluded_count = _count_csv_rows(exclusion_path)
            width, height, dtype = _image_dimensions(
                run_logger.data.get("events", [])[event_start:]
            )
            row: dict[str, Any] = {
                "file_name": image_path.name,
                "file_path": str(image_path),
                "file_type": image_path.suffix.lower().lstrip(".").upper(),
                "status": "success",
                "image_width_px": width,
                "image_height_px": height,
                "image_dtype": dtype,
                "sunspot_count": sunspot_count,
                "excluded_sunspot_count": excluded_count,
                "statistics_csv": str(statistics_path),
                "exclusion_list_csv": str(exclusion_path),
                "output_directory": str(output_dir),
                "error": "",
            }
            run_logger.record_event("batch_image_completed", row)
            LOGGER.info(
                "一括解析完了: %s | 検出=%d | 除外=%d",
                image_path.name,
                sunspot_count,
                excluded_count,
            )
        except Exception as error:
            row = {
                "file_name": image_path.name,
                "file_path": str(image_path),
                "file_type": image_path.suffix.lower().lstrip(".").upper(),
                "status": "failed",
                "image_width_px": "",
                "image_height_px": "",
                "image_dtype": "",
                "sunspot_count": "",
                "excluded_sunspot_count": "",
                "statistics_csv": "",
                "exclusion_list_csv": "",
                "output_directory": str(output_dir),
                "error": f"{type(error).__name__}: {error}",
            }
            run_logger.record_event(
                "batch_image_failed",
                {
                    "file_name": image_path.name,
                    "file_path": str(image_path),
                    "output_directory": str(output_dir),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            LOGGER.exception("一括解析失敗: %s", image_path.name)

        rows.append(row)
        _write_summary(summary_path, rows)

    success_count = sum(row["status"] == "success" for row in rows)
    failed_count = len(rows) - success_count
    overall_status = (
        "failed"
        if success_count == 0
        else "partial_success"
        if failed_count
        else "success"
    )
    summary = {
        "overall_status": overall_status,
        "input_folder": str(image_files[0].parent.resolve()),
        "total_image_count": len(rows),
        "successful_image_count": success_count,
        "failed_image_count": failed_count,
        "summary_csv": str(summary_path),
        "images_directory": str(image_output_root),
        "images": rows,
    }
    run_logger.record_event(
        "batch_analysis_completed",
        {key: value for key, value in summary.items() if key != "images"},
    )
    return summary
