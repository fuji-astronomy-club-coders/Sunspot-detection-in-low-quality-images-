"""Validate and execute same-day multi-observatory sunspot comparisons."""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.main import process_image
from src.package import limb_darking_resolution as ldr
from src.package import utils
from src.package.config import PROJECT_ROOT
from src.package.run_logger import RunLogger
from src.package.solar_coordinates import Orientation, SolarGeometry, SolarTransform

LOGGER = logging.getLogger("sunspot.comparison")
SOURCE_INPUT_ROOT = PROJECT_ROOT / "source_input"
IMAGE_SUFFIXES = {".tif", ".tiff", ".fits", ".fit", ".fts"}
SOURCE_KEYS = {"source_id", "display_name", "orientation_policy", "north_vector", "flip_x"}
SAMPLE_KEYS = {
    "source_id",
    "image",
    "pixel_scale_arcsec_per_pixel",
    "north_vector",
    "flip_x",
}


class ComparisonInputError(ValueError):
    """Raised when the comparison set cannot be started safely."""


@dataclass(frozen=True, slots=True)
class PreparedSample:
    """One validated (or individually skippable) source sample."""

    source_id: str | None
    source_name: str | None
    image_path: Path | None
    pixel_scale: float | None
    orientation: Orientation | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ComparisonPlan:
    """Validated input files and source definitions for one comparison date."""

    date_folder: Path
    comparison_date: str
    sources: tuple[dict[str, Any], ...]
    samples: tuple[PreparedSample, ...]
    mask_template_path: Path | None
    normalized_polygons: tuple[list[list[float]], ...] | None


def _load_yaml(path: Path) -> object:
    """Load YAML while converting parser errors into contextual input errors."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as error:
        raise ComparisonInputError(f"{path.name} を読み込めません: {error}") from error


def _parse_north_vector(value: object) -> tuple[float, float]:
    """Validate and normalize a YAML north-vector mapping."""
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        raise ValueError("north_vector は x と y を持つ辞書で指定してください。")
    try:
        x, y = float(value["x"]), float(value["y"])
    except (TypeError, ValueError) as error:
        raise ValueError("north_vector の x と y は数値にしてください。") from error
    if not math.isfinite(x) or not math.isfinite(y) or math.hypot(x, y) == 0:
        raise ValueError("north_vector は有限の非ゼロベクトルにしてください。")
    return x, y


def _parse_orientation(value: dict[str, Any]) -> Orientation:
    """Create an orientation after validating both required per-image fields."""
    if "north_vector" not in value or "flip_x" not in value:
        raise ValueError("north_vector と flip_x の両方を指定してください。")
    if not isinstance(value["flip_x"], bool):
        raise ValueError("flip_x は true または false にしてください。")
    return Orientation(
        north_vector=_parse_north_vector(value["north_vector"]),
        flip_x=value["flip_x"],
    )


def _load_mask_template(path: Path) -> tuple[list[list[list[float]]], dict[str, Any]]:
    """Validate a normalized-coordinate polygon template."""
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonInputError(f"マスクテンプレートを読み込めません: {error}") from error

    if not isinstance(template, dict):
        raise ComparisonInputError("マスクテンプレートのルートは JSON object にしてください。")
    if template.get("schema_version") != 1:
        raise ComparisonInputError("未対応のマスクテンプレート schema_version です。")
    if template.get("coordinate_system") != "solar_disk_normalized":
        raise ComparisonInputError("マスクテンプレートの座標系が solar_disk_normalized ではありません。")
    regions = template.get("regions")
    if not isinstance(regions, list):
        raise ComparisonInputError("マスクテンプレートの regions は配列にしてください。")

    polygons: list[list[list[float]]] = []
    for index, region in enumerate(regions, start=1):
        if not isinstance(region, dict) or not isinstance(region.get("vertices"), list):
            raise ComparisonInputError(f"マスク領域 {index} の vertices がありません。")
        vertices = region["vertices"]
        if len(vertices) < 3:
            raise ComparisonInputError(f"マスク領域 {index} は3頂点以上必要です。")
        polygon: list[list[float]] = []
        for point in vertices:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ComparisonInputError(f"マスク領域 {index} の頂点形式が不正です。")
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError) as error:
                raise ComparisonInputError(f"マスク領域 {index} の頂点は数値にしてください。") from error
            if not math.isfinite(x) or not math.isfinite(y):
                raise ComparisonInputError(f"マスク領域 {index} の頂点には有限値が必要です。")
            polygon.append([x, y])
        polygons.append(polygon)
    return polygons, template


def prepare_comparison(date_folder: str | Path) -> ComparisonPlan:
    """Validate all comparison-level inputs before creating a run session.

    Invalid individual sample rows are retained as skippable entries. A missing
    profile, invalid date directory, or invalid shared mask template aborts setup.
    """
    selected_folder = Path(date_folder).expanduser()
    if not selected_folder.is_absolute():
        selected_folder = PROJECT_ROOT / selected_folder
    selected_folder = selected_folder.resolve()
    source_root = SOURCE_INPUT_ROOT.resolve()

    if not selected_folder.is_relative_to(source_root) or selected_folder == source_root:
        raise ComparisonInputError(
            f"比較フォルダーは {source_root} 配下の日付フォルダーを指定してください。"
        )
    if not selected_folder.is_dir():
        raise ComparisonInputError(f"日付フォルダーがありません: {selected_folder}")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", selected_folder.name) is None:
        raise ComparisonInputError("日付フォルダー名は YYYY-MM-DD 形式にしてください。")
    try:
        comparison_date = date.fromisoformat(selected_folder.name).isoformat()
    except ValueError as error:
        raise ComparisonInputError("日付フォルダー名は実在する YYYY-MM-DD にしてください。") from error

    sources_path = source_root / "sources.yaml"
    if not sources_path.is_file():
        raise ComparisonInputError(
            f"{sources_path} がありません。docs/templates/sources.example.yaml をコピーしてください。"
        )
    sources_document = _load_yaml(sources_path)
    if not isinstance(sources_document, dict) or set(sources_document) != {"sources"}:
        raise ComparisonInputError("sources.yaml には sources のみを指定してください。")
    raw_sources = sources_document["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != 4:
        raise ComparisonInputError("sources.yaml にはちょうど4観測所を登録してください。")

    sources: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for index, raw_source in enumerate(raw_sources, start=1):
        if not isinstance(raw_source, dict):
            raise ComparisonInputError(f"sources.yaml の {index} 番目の観測所設定が不正です。")
        unknown = set(raw_source) - SOURCE_KEYS
        if unknown:
            raise ComparisonInputError(
                f"sources.yaml に未対応のキーがあります: {', '.join(sorted(unknown))}"
            )
        source_id = raw_source.get("source_id")
        display_name = raw_source.get("display_name")
        policy = raw_source.get("orientation_policy")
        if (
            not isinstance(source_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", source_id) is None
        ):
            raise ComparisonInputError(
                f"{index} 番目の観測所 source_id は英数字・ハイフン・アンダースコアで指定してください。"
            )
        if source_id in source_ids:
            raise ComparisonInputError(f"source_id が重複しています: {source_id}")
        source_ids.add(source_id)
        if not isinstance(display_name, str) or not display_name.strip():
            raise ComparisonInputError(f"{source_id} の display_name が必要です。")
        if policy not in ("fixed", "per_sample"):
            raise ComparisonInputError(f"{source_id} の orientation_policy が不正です。")

        normalized_source = dict(raw_source)
        normalized_source["source_id"] = source_id
        normalized_source["display_name"] = display_name
        if policy == "per_sample" and (
            "north_vector" in raw_source or "flip_x" in raw_source
        ):
            raise ComparisonInputError(
                f"{source_id}: per_sample の方向は comparison.yaml に指定してください。"
            )
        if policy == "fixed":
            try:
                orientation = _parse_orientation(raw_source)
            except ValueError as error:
                raise ComparisonInputError(f"{source_id}: {error}") from error
            normalized_source["north_vector"] = {
                "x": orientation.north_vector[0],
                "y": orientation.north_vector[1],
            }
            normalized_source["flip_x"] = orientation.flip_x
        sources.append(normalized_source)

    comparison_path = selected_folder / "comparison.yaml"
    if not comparison_path.is_file():
        raise ComparisonInputError(f"{comparison_path} がありません。")
    comparison_document = _load_yaml(comparison_path)
    if not isinstance(comparison_document, dict):
        raise ComparisonInputError("comparison.yaml のルートは YAML mapping にしてください。")
    unknown_top_level = set(comparison_document) - {"mask_template", "samples"}
    if unknown_top_level:
        raise ComparisonInputError(
            "comparison.yaml に未対応のトップレベルキーがあります: "
            + ", ".join(sorted(unknown_top_level))
        )
    raw_samples = comparison_document.get("samples")
    if not isinstance(raw_samples, list) or not 1 <= len(raw_samples) <= 4:
        raise ComparisonInputError("samples には1～4観測所を指定してください。")

    mask_template_value = comparison_document.get("mask_template")
    mask_template_path: Path | None = None
    normalized_polygons: tuple[list[list[float]], ...] | None = None
    if mask_template_value is not None:
        if not isinstance(mask_template_value, str) or not mask_template_value.strip():
            raise ComparisonInputError("mask_template は相対パス文字列にしてください。")
        candidate_template = Path(mask_template_value)
        if candidate_template.is_absolute():
            raise ComparisonInputError("mask_template は日付フォルダーからの相対パスにしてください。")
        mask_template_path = (selected_folder / candidate_template).resolve()
        if not mask_template_path.is_relative_to(source_root):
            raise ComparisonInputError("mask_template は source_input/ 配下を指定してください。")
        if not mask_template_path.is_file():
            raise ComparisonInputError(f"mask_template がありません: {mask_template_path}")
        polygons, _template = _load_mask_template(mask_template_path)
        normalized_polygons = tuple(polygons)

    source_map = {source["source_id"]: source for source in sources}
    raw_ids = [
        row.get("source_id")
        for row in raw_samples
        if isinstance(row, dict) and isinstance(row.get("source_id"), str)
    ]
    duplicate_ids = {source_id for source_id in raw_ids if raw_ids.count(source_id) > 1}
    prepared_samples: list[PreparedSample] = []

    for index, raw_sample in enumerate(raw_samples, start=1):
        if not isinstance(raw_sample, dict):
            prepared_samples.append(
                PreparedSample(None, None, None, None, None, f"samples の {index} 番目は mapping にしてください。")
            )
            continue

        source_id = raw_sample.get("source_id")
        source = source_map.get(source_id) if isinstance(source_id, str) else None
        source_name = source["display_name"] if source else None
        error: str | None = None

        if set(raw_sample) - SAMPLE_KEYS:
            error = "未対応のサンプルキーがあります。"
        elif not isinstance(source_id, str) or source is None:
            error = f"未登録の source_id です: {source_id}"
        elif source_id in duplicate_ids:
            error = f"source_id が同じ日付内で重複しています: {source_id}"
        elif not isinstance(raw_sample.get("image"), str) or not raw_sample["image"].strip():
            error = "image に日付フォルダーからの相対パスが必要です。"

        image_path: Path | None = None
        if error is None:
            relative_image = Path(raw_sample["image"])
            if relative_image.is_absolute():
                error = "image は日付フォルダーからの相対パスにしてください。"
            else:
                resolved_image = (selected_folder / relative_image).resolve()
                if not resolved_image.is_relative_to(selected_folder):
                    error = "image が日付フォルダーの外を参照しています。"
                elif resolved_image.suffix.lower() not in IMAGE_SUFFIXES:
                    error = "対応形式は tif / tiff / fits / fit / fts です。"
                elif not resolved_image.is_file():
                    error = f"画像ファイルがありません: {raw_sample['image']}"
                else:
                    image_path = resolved_image
                    try:
                        probe_image = utils.loadImage(resolved_image)
                        if probe_image.ndim != 2:
                            error = "比較モードでは2次元グレースケール画像のみ対応します。"
                        del probe_image
                    except Exception as image_error:
                        error = f"画像を読み込めません: {image_error}"

        pixel_scale: float | None = None
        if error is None:
            try:
                pixel_scale = float(raw_sample["pixel_scale_arcsec_per_pixel"])
            except (KeyError, TypeError, ValueError):
                error = "pixel_scale_arcsec_per_pixel に正の数値が必要です。"
            else:
                if not math.isfinite(pixel_scale) or pixel_scale <= 0:
                    error = "pixel_scale_arcsec_per_pixel に正の有限値が必要です。"

        orientation: Orientation | None = None
        if error is None and source is not None:
            if source["orientation_policy"] == "fixed":
                if "north_vector" in raw_sample or "flip_x" in raw_sample:
                    error = "fixed の観測所では日付別の north_vector / flip_x を指定できません。"
                else:
                    orientation = _parse_orientation(source)
            else:
                try:
                    orientation = _parse_orientation(raw_sample)
                except ValueError as orientation_error:
                    error = str(orientation_error)

        prepared_samples.append(
            PreparedSample(
                source_id if isinstance(source_id, str) else None,
                source_name,
                image_path,
                pixel_scale,
                orientation,
                error,
            )
        )

    return ComparisonPlan(
        selected_folder,
        comparison_date,
        tuple(sources),
        tuple(prepared_samples),
        mask_template_path,
        normalized_polygons,
    )


def _write_mask_template(
    plan: ComparisonPlan,
    source_id: str,
    image_path: Path,
    geometry: SolarGeometry,
    orientation: Orientation,
    polygons: list[list[float]],
) -> Path:
    """Persist the selected base-image mask in normalized solar coordinates."""
    target_folder = SOURCE_INPUT_ROOT / "mask_templates"
    target_folder.mkdir(parents=True, exist_ok=True)
    target_path = target_folder / f"{plan.comparison_date}_{source_id}.json"
    document = {
        "schema_version": 1,
        "coordinate_system": "solar_disk_normalized",
        "created_from": {
            "date": plan.comparison_date,
            "source_id": source_id,
            "image": image_path.name,
            "solar_geometry": {
                "center_x_px": geometry.center[0],
                "center_y_px": geometry.center[1],
                "solar_radius_px": geometry.radius,
            },
            "north_vector": {
                "x": orientation.north_vector[0],
                "y": orientation.north_vector[1],
            },
            "flip_x": orientation.flip_x,
        },
        "regions": [
            {"id": index, "vertices": polygon}
            for index, polygon in enumerate(polygons, start=1)
        ],
    }
    target_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_path


def _augment_csv(path: Path, transform: SolarTransform) -> tuple[list[str], list[dict[str, str]]]:
    """Append normalized centroid coordinates to one source CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"解析結果 CSV がありません: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for column in ("centroid_x_normalized", "centroid_y_normalized"):
        if column not in fieldnames:
            fieldnames.append(column)

    for row in rows:
        try:
            pixel_x = float(row["centroid_x_px"])
            pixel_y = float(row["centroid_y_px"])
            normalized_x, normalized_y = transform.pixel_to_normalized(pixel_x, pixel_y)
            row["centroid_x_normalized"] = f"{normalized_x:.6f}"
            row["centroid_y_normalized"] = f"{normalized_y:.6f}"
        except (KeyError, TypeError, ValueError):
            row["centroid_x_normalized"] = ""
            row["centroid_y_normalized"] = ""

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)
    return fieldnames, rows


def _combine_csv(
    session_dir: Path,
    file_name: str,
    sources: list[dict[str, Any]],
) -> None:
    """Combine augmented per-source CSVs while preserving their original columns."""
    source_rows: list[tuple[dict[str, Any], dict[str, str]]] = []
    fieldnames: list[str] = ["source_id", "source_name", "image"]
    for source in sources:
        source_file = Path(source["output_dir"]) / file_name
        with source_file.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            current_fields = list(reader.fieldnames or [])
            for field in current_fields:
                if field not in fieldnames:
                    fieldnames.append(field)
            source_rows.extend((source, row) for row in reader)

    output_path = session_dir / f"comparison_{file_name}"
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for source, row in source_rows:
            writer.writerow(
                {
                    "source_id": source["source_id"],
                    "source_name": source["source_name"],
                    "image": source["image"],
                    **row,
                }
            )


def run_comparison(
    plan: ComparisonPlan,
    base_config: dict[str, Any],
    run_logger: RunLogger,
) -> dict[str, Any]:
    """Process available source images and create the combined comparison outputs."""
    result_rows: list[dict[str, Any]] = []
    successful_sources: list[dict[str, Any]] = []
    supplied_ids = {
        sample.source_id for sample in plan.samples if sample.source_id is not None
    }
    source_map = {source["source_id"]: source for source in plan.sources}
    normalized_polygons = (
        [list(map(list, polygon)) for polygon in plan.normalized_polygons]
        if plan.normalized_polygons is not None
        else None
    )
    generated_mask_path: Path | None = None

    for source_id, source in source_map.items():
        if source_id not in supplied_ids:
            result_rows.append(
                {
                    "source_id": source_id,
                    "source_name": source["display_name"],
                    "status": "not_provided",
                    "reason": "comparison.yaml にサンプルがありません。",
                }
            )

    for sample_index, sample in enumerate(plan.samples, start=1):
        if sample.error is not None:
            result_rows.append(
                {
                    "source_id": sample.source_id,
                    "source_name": sample.source_name,
                    "status": "skipped_invalid_input",
                    "reason": sample.error,
                }
            )
            run_logger.record_event(
                "comparison_sample_skipped",
                {"sample_index": sample_index, "source_id": sample.source_id, "reason": sample.error},
            )
            continue

        assert sample.source_id is not None
        assert sample.source_name is not None
        assert sample.image_path is not None
        assert sample.pixel_scale is not None
        assert sample.orientation is not None

        source_output = run_logger.session_dir / sample.source_id
        source_output.mkdir(parents=True, exist_ok=True)
        source_started_event_count = len(run_logger.data["events"])

        try:
            image = utils.loadImage(sample.image_path)
            if image.ndim != 2:
                raise ValueError("比較モードでは2次元グレースケール画像のみ対応します。")
            geometry_tuple = ldr.find_solar_geometry(image)
            geometry = SolarGeometry(geometry_tuple[0], geometry_tuple[1])
            transform = SolarTransform(geometry, sample.orientation)
            angular_radius = geometry.radius * sample.pixel_scale
            angular_warning = not 900.0 <= angular_radius <= 1000.0
            if angular_warning:
                LOGGER.warning(
                    "%s の太陽角半径が暫定範囲外です: %.2f arcsec",
                    sample.source_id,
                    angular_radius,
                )

            sample_config = copy.deepcopy(base_config)
            sample_config["Paths"]["input_image"] = str(sample.image_path)
            sample_config["Display"] = dict(sample_config.get("Display") or {})

            pixel_polygons: list[np.ndarray] | None = None
            if normalized_polygons is not None:
                pixel_polygons = [
                    transform.normalized_points_to_pixel(polygon)
                    for polygon in normalized_polygons
                ]

            run_logger.record_event(
                "comparison_sample_started",
                {
                    "source_id": sample.source_id,
                    "source_name": sample.source_name,
                    "image": str(sample.image_path),
                    "pixel_scale_arcsec_per_pixel": sample.pixel_scale,
                    "solar_angular_radius_arcsec": angular_radius,
                    "angular_radius_warning": angular_warning,
                    "orientation": transform.as_dict(),
                    "mask_template": str(plan.mask_template_path or generated_mask_path)
                    if plan.mask_template_path or generated_mask_path
                    else None,
                },
            )

            spot_count = process_image(
                config=sample_config,
                run_logger=run_logger,
                output_dir=source_output,
                solar_geometry_override=geometry_tuple,
                mask_polygons=pixel_polygons,
            )

            if normalized_polygons is None:
                selection_events = [
                    event
                    for event in run_logger.data["events"][source_started_event_count:]
                    if event["event"] == "mask_regions_selected"
                ]
                if not selection_events:
                    raise RuntimeError("基準画像の共通マスク選択イベントを取得できませんでした。")
                selected_regions = selection_events[-1]["details"].get("regions", [])
                normalized_polygons = [
                    transform.rectangle_to_normalized_polygon(
                        (
                            float(region["x"]),
                            float(region["y"]),
                            float(region["width"]),
                            float(region["height"]),
                        )
                    )
                    for region in selected_regions
                ]
                generated_mask_path = _write_mask_template(
                    plan,
                    sample.source_id,
                    sample.image_path,
                    geometry,
                    sample.orientation,
                    normalized_polygons,
                )
                run_logger.record_event(
                    "comparison_mask_template_created",
                    {
                        "source_id": sample.source_id,
                        "path": str(generated_mask_path),
                        "region_count": len(normalized_polygons),
                    },
                )

            successful_sources.append(
                {
                    "source_id": sample.source_id,
                    "source_name": sample.source_name,
                    "image": sample.image_path.name,
                    "image_path": str(sample.image_path),
                    "pixel_scale_arcsec_per_pixel": sample.pixel_scale,
                    "solar_geometry": {
                        "center_x_px": geometry.center[0],
                        "center_y_px": geometry.center[1],
                        "radius_px": geometry.radius,
                        "angular_radius_arcsec": angular_radius,
                    },
                    "orientation": transform.as_dict(),
                    "transform": transform,
                    "output_dir": str(source_output),
                    "sunspot_count": int(spot_count),
                    "mask_region_count": len(normalized_polygons or []),
                    "mask_template": str(plan.mask_template_path or generated_mask_path)
                    if plan.mask_template_path or generated_mask_path
                    else None,
                }
            )
            result_rows.append(
                {
                    "source_id": sample.source_id,
                    "source_name": sample.source_name,
                    "status": "success",
                    "sunspot_count": int(spot_count),
                    "output_dir": str(source_output),
                    "solar_angular_radius_arcsec": angular_radius,
                    "angular_radius_warning": angular_warning,
                }
            )
            run_logger.record_event(
                "comparison_sample_completed",
                {"source_id": sample.source_id, "sunspot_count": int(spot_count)},
            )
        except Exception as error:
            LOGGER.exception("比較サンプル %s の処理に失敗しました。", sample.source_id)
            result_rows.append(
                {
                    "source_id": sample.source_id,
                    "source_name": sample.source_name,
                    "status": "failed_processing",
                    "reason": str(error),
                }
            )
            run_logger.record_event(
                "comparison_sample_failed",
                {
                    "source_id": sample.source_id,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
            )

    successful_ids = {source["source_id"] for source in successful_sources}
    all_success = len(successful_ids) == 4
    if all_success:
        overall_status = "success"
    elif successful_sources:
        overall_status = "partial_success"
    else:
        overall_status = "failed"

    if successful_sources:
        for source in successful_sources:
            source_transform = source.pop("transform")
            output_dir = Path(source["output_dir"])
            for csv_name in ("sunspot_statistics.csv", "sunspot_exclusion_list.csv"):
                _augment_csv(output_dir / csv_name, source_transform)
        _combine_csv(
            run_logger.session_dir,
            "sunspot_statistics.csv",
            successful_sources,
        )
        _combine_csv(
            run_logger.session_dir,
            "sunspot_exclusion_list.csv",
            successful_sources,
        )

    summary = {
        "comparison_date": plan.comparison_date,
        "registered_sources": [
            {
                "source_id": source["source_id"],
                "display_name": source["display_name"],
                "orientation_policy": source["orientation_policy"],
            }
            for source in plan.sources
        ],
        "samples": result_rows,
        "successful_sources": successful_sources,
        "mask_template": str(plan.mask_template_path or generated_mask_path)
        if plan.mask_template_path or generated_mask_path
        else None,
        "overall_status": overall_status,
        "output_directory": str(run_logger.session_dir),
    }
    summary_path = run_logger.session_dir / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    run_logger.record_event(
        "comparison_completed",
        {
            "overall_status": overall_status,
            "successful_sample_count": len(successful_sources),
            "sample_count": len(plan.samples),
            "summary_path": str(summary_path),
        },
    )
    return summary
