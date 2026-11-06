"""プロジェクト全体で使用する共通ユーティリティを提供するモジュール。

複数のモジュールから再利用される汎用的な処理を集約する。
特定の機能や実行モードに依存する処理は ``src/package/`` 配下に配置し、
本モジュールには共通性の高い処理のみを実装する。

アプリケーションのメイン処理は ``src/main.py`` に、
実行制御はプロジェクトルートの ``run.py`` に配置する。
"""
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tifffile
from astropy.io import fits
from matplotlib import font_manager
from PIL import Image as PlImage
from PIL import PngImagePlugin

LOGGER = logging.getLogger("sunspot.utils")


def loadImage(image_path: str | Path) -> npt.NDArray[Any]:
    """TIFFまたはFITS画像を読み込み、画素値をNumPy配列で返す。

    Parameters
    ----------
    image_path : str | pathlib.Path
        読み込む画像ファイルのパス。

    Returns
    -------
    numpy.ndarray
        各画素の値を格納した配列。

    Raises
    ------
    ValueError
        対応していない画像形式または存在しないファイルが指定された場合。
    """
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"存在しないパスです: {path}")

    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        image = tifffile.imread(path)
    elif suffix in (".fits", ".fit", ".fts"):
        image = fits.getdata(path)
    else:
        raise ValueError(f"対応していない画像形式です: {suffix}")
    if image is None:
        raise ValueError(f"failed to load the image: {image_path}")
    
    # NaNを光度0に置き換える
    image = np.nan_to_num(image, nan=0.0)
    
    return np.asarray(image)


type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]


def _metadata_to_json(value: object, max_items: int = 256) -> JSONValue:
    """Convert metadata values to bounded, JSON-compatible Python values."""
    if value is None or isinstance(value, (str, int, bool)):
        if isinstance(value, str) and len(value) > 65536:
            return value[:65536] + f"... [truncated {len(value) - 65536} characters]"
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _metadata_to_json(value.item(), max_items=max_items)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _metadata_to_json(value.item(), max_items=max_items)
        flat = value.reshape(-1)
        preview = [
            _metadata_to_json(item, max_items=max_items)
            for item in flat[:max_items]
        ]
        if flat.size > max_items:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "preview": preview,
                "omitted_items": int(flat.size - max_items),
            }
        return preview
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "hex": value[:max_items].hex(),
                "length_bytes": len(value),
                "truncated": len(value) > max_items,
            }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _metadata_to_json(item, max_items=max_items)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        converted = [
            _metadata_to_json(item, max_items=max_items)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            converted.append(f"... [{len(value) - max_items} more items]")
        return converted
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def collect_image_metadata(
    image_path: str | Path,
    image: np.ndarray,
) -> dict[str, Any]:
    """Collect source-file, decoded-array, and format-specific image metadata.

    TIFF metadata includes the first 32 pages' tags and embedded ImageJ/OME
    descriptions. FITS metadata includes every HDU header. Pixel data is not
    copied into the log.
    """
    path = Path(image_path)
    stat = path.stat()
    image_info: dict[str, Any] = {
        "path": str(path.resolve()),
        "file_name": path.name,
        "file_extension": path.suffix.lower(),
        "file_size_bytes": int(stat.st_size),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "array": {
            "shape": [int(dimension) for dimension in image.shape],
            "dimensions": int(image.ndim),
            "dtype": str(image.dtype),
            "pixel_count": int(image.size),
            "array_size_bytes": int(image.nbytes),
        },
    }
    if image.ndim == 2:
        image_info["array"]["height_px"] = int(image.shape[0])
        image_info["array"]["width_px"] = int(image.shape[1])
    elif image.ndim >= 3 and image.shape[-1] in (3, 4):
        # TIFF RGB/RGBA arrays commonly use YXS or YXC axis order.
        image_info["array"]["height_px"] = int(image.shape[-3])
        image_info["array"]["width_px"] = int(image.shape[-2])
        image_info["array"]["channel_count"] = int(image.shape[-1])
        image_info["array"]["leading_dimensions"] = [
            int(dimension) for dimension in image.shape[:-3]
        ]
    elif image.ndim >= 3:
        image_info["array"]["height_px"] = int(image.shape[-2])
        image_info["array"]["width_px"] = int(image.shape[-1])
        image_info["array"]["leading_dimensions"] = [
            int(dimension) for dimension in image.shape[:-2]
        ]

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tiff:
            series = [
                {
                    "index": index,
                    "shape": [int(dimension) for dimension in item.shape],
                    "dtype": str(item.dtype),
                    "axes": str(item.axes),
                    "kind": str(item.kind),
                }
                for index, item in enumerate(tiff.series)
            ]
            pages = []
            for page_index, page in enumerate(tiff.pages[:32]):
                tags = {}
                # tifffile may expose non-keyframe pages as TiffFrame objects.
                # Resolve their keyframe to access the shared TIFF tag collection.
                tag_page = cast(tifffile.TiffPage, getattr(page, "keyframe", page))
                for tag in tag_page.tags.values():
                    tags[str(tag.name)] = _metadata_to_json(tag.value)
                pages.append(
                    {
                        "index": page_index,
                        "shape": [int(dimension) for dimension in page.shape],
                        "dtype": str(page.dtype),
                        "tags": tags,
                    }
                )
            image_info["container_metadata"] = {
                "format": "TIFF",
                "byte_order": str(tiff.byteorder),
                "is_bigtiff": bool(tiff.is_bigtiff),
                "page_count": len(tiff.pages),
                "series": series,
                "pages": pages,
                "omitted_page_metadata_count": max(0, len(tiff.pages) - len(pages)),
                "imagej_metadata": _metadata_to_json(tiff.imagej_metadata),
                "ome_metadata": _metadata_to_json(tiff.ome_metadata),
            }
    elif suffix in (".fits", ".fit", ".fts"):
        hdu_metadata = []
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            for hdu_index, hdu in enumerate(hdul):
                header_metadata: dict[str, Any] = {}
                for card in hdu.header.cards:
                    keyword = str(card.keyword).strip()
                    if not keyword:
                        continue
                    card_value = {
                        "value": _metadata_to_json(card.value),
                        "comment": str(card.comment),
                    }
                    if keyword in header_metadata:
                        current = header_metadata[keyword]
                        if not isinstance(current, list):
                            current = [current]
                        current.append(card_value)
                        header_metadata[keyword] = current
                    else:
                        header_metadata[keyword] = card_value

                naxis = int(hdu.header.get("NAXIS", 0) or 0)
                fits_shape = [
                    int(hdu.header.get(f"NAXIS{axis}", 0) or 0)
                    for axis in range(naxis, 0, -1)
                ]
                hdu_metadata.append(
                    {
                        "index": hdu_index,
                        "name": str(hdu.name),
                        "type": type(hdu).__name__,
                        "shape": fits_shape,
                        "header": header_metadata,
                    }
                )
        image_info["container_metadata"] = {
            "format": "FITS",
            "hdu_count": len(hdu_metadata),
            "hdus": hdu_metadata,
        }
    else:
        image_info["container_metadata"] = {
            "format": suffix.lstrip(".").upper() or "UNKNOWN",
            "format_specific_metadata": "No format-specific reader is configured.",
        }

    converted = _metadata_to_json(image_info)
    if not isinstance(converted, dict):
        raise TypeError("画像メタデータのルートは辞書である必要があります。")
    return cast(dict[str, Any], converted)

def svImage_withInfo(image: np.ndarray, info: dict | str, path: str | Path) -> None:
    """画像をメタデータ(PngInfo)とともに保存します。

    Parameters
    ----------
    image : np.ndarray
        保存する画像データ(np.ndarray)。
    info : dict | str
        画像に付与するメタデータやパラメータ。
        辞書型の場合は各キー・バリューがPNGのテキストチャンクとして保存されます。
        文字列の場合は {"info": 文字列} として保存されます。
    path : str | Path
        保存先のファイルパス。
    
    Returns
    -------
    None
    """
    file_path = Path(path)
    # 保存先の親ディレクトリを作成
    file_path.parent.mkdir(parents=True, exist_ok=True)

    im = PlImage.fromarray(image)

    # メタデータ（PngInfo）を作成
    metadata = PngImagePlugin.PngInfo()
    valid_info = info if isinstance(info, dict) else {"info": info}

    for key, value in valid_info.items():
        # 辞書型の場合はJSON文字列に変換、それ以外は文字列に変換
        param_str = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
        
        # キーと値のペアでパラメータを追加
        metadata.add_text(key, param_str)

    try:
        # パラメータを含めて保存
        im.save(file_path, pnginfo=metadata)
        LOGGER.info("画像を保存しました: %s", file_path)
    except Exception:
        LOGGER.exception("画像の保存に失敗しました: %s", file_path)


def save_sunspot_statistics_csv(
    label_image: np.ndarray,
    excluded_labels: list[int],
    output_path: str | Path,
) -> int:
    """マスク適用後に残った黒点の重心・外接矩形・面積を CSV に保存する。

    Parameters
    ----------
    label_image : numpy.ndarray
        黒点ごとに異なる正の整数ラベルを持つ2次元画像。
    excluded_labels : list[int]
        マスク領域によって除外されたラベル。
    output_path : str | pathlib.Path
        CSV の保存先。

    Returns
    -------
    int
        CSV に出力した黒点数。
    """
    if label_image.ndim != 2:
        raise ValueError("黒点ラベル画像は2次元配列にしてください。")

    retained_labels = label_image.copy()
    if excluded_labels:
        retained_labels[np.isin(retained_labels, excluded_labels)] = 0

    spot_labels = np.unique(retained_labels)
    spot_labels = spot_labels[spot_labels > 0]
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    field_names = [
        "sunspot_id",
        "label",
        "centroid_x_px",
        "centroid_y_px",
        "bbox_x_px",
        "bbox_y_px",
        "bbox_width_px",
        "bbox_height_px",
        "area_px",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(field_names)
        for sunspot_id, spot_label in enumerate(spot_labels, start=1):
            y_coordinates, x_coordinates = np.nonzero(
                retained_labels == spot_label
            )
            min_x = int(x_coordinates.min())
            max_x = int(x_coordinates.max())
            min_y = int(y_coordinates.min())
            max_y = int(y_coordinates.max())
            writer.writerow(
                [
                    sunspot_id,
                    int(spot_label),
                    round(float(x_coordinates.mean()), 6),
                    round(float(y_coordinates.mean()), 6),
                    min_x,
                    min_y,
                    max_x - min_x + 1,
                    max_y - min_y + 1,
                    int(x_coordinates.size),
                ]
            )

    LOGGER.info(
        "黒点統計 CSV を保存しました: %s (黒点数: %d)",
        csv_path,
        len(spot_labels),
    )
    LOGGER.debug(
        "CSV 座標系: 左上原点、0始まり。外接矩形サイズは幅・高さ、面積は画素数。",
    )
    return int(len(spot_labels))


def save_sunspot_bounding_box_png(
    background_image: np.ndarray,
    label_image: np.ndarray,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Save a diagnostic PNG with each retained component's bbox and CSV label.

    The background uses percentile normalization for consistent 8-bit display,
    including when the source image is 16-bit or floating-point. Bounding boxes
    and their labels are derived from the same label image used in the
    sunspot_statistics.csv label column.

    Parameters
    ----------
    background_image : numpy.ndarray
        Two-dimensional emphasized image used for component extraction.
    label_image : numpy.ndarray
        Two-dimensional integer label image after mask exclusions.
    output_path : str | pathlib.Path
        Destination PNG path.
    metadata : dict[str, Any] | None, optional
        Additional PNG metadata.

    Returns
    -------
    int
        Number of labeled components drawn.
    """
    if background_image.ndim != 2 or label_image.ndim != 2:
        raise ValueError("背景画像と黒点ラベル画像は2次元配列にしてください。")
    if background_image.shape != label_image.shape:
        raise ValueError("背景画像と黒点ラベル画像のサイズが一致していません。")

    source = np.asarray(background_image, dtype=np.float64)
    finite_mask = np.isfinite(source)
    gray_image = np.zeros(source.shape, dtype=np.uint8)
    if np.any(finite_mask):
        finite_values = source[finite_mask]
        low_value, high_value = np.percentile(finite_values, [1.0, 99.5])
        if high_value > low_value:
            normalized = np.zeros(source.shape, dtype=np.float64)
            normalized[finite_mask] = (
                (finite_values - low_value) * 255.0 / (high_value - low_value)
            )
            gray_image = np.clip(normalized, 0.0, 255.0).astype(np.uint8)

    canvas = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    height, width = label_image.shape
    labels = np.unique(label_image)
    labels = labels[labels > 0]
    line_width = max(1, int(round(min(height, width) / 800)))
    font_scale = max(0.35, min(0.9, min(height, width) / 1800.0))
    text_thickness = max(1, line_width)

    for label_value in labels:
        y_coordinates, x_coordinates = np.nonzero(label_image == label_value)
        if x_coordinates.size == 0:
            continue

        x1 = int(x_coordinates.min())
        x2 = int(x_coordinates.max())
        y1 = int(y_coordinates.min())
        y2 = int(y_coordinates.max())
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            line_width,
            lineType=cv2.LINE_AA,
        )

        label_text = str(int(label_value))
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        text_x = min(max(0, x1), max(0, width - text_width - 4))
        if y1 >= text_height + baseline + 6:
            text_baseline = y1 - 4
        else:
            text_baseline = min(height - baseline - 1, y1 + text_height + baseline + 5)
        text_top = max(0, text_baseline - text_height - baseline)
        text_right = min(width - 1, text_x + text_width + 4)
        text_bottom = min(height - 1, text_baseline + baseline)
        cv2.rectangle(
            canvas,
            (text_x, text_top),
            (text_right, text_bottom),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            canvas,
            label_text,
            (text_x + 2, text_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    output_metadata: dict[str, Any] = {
        "label_mapping": {
            "csv_file": "sunspot_statistics.csv",
            "csv_column": "label",
            "labels": [int(label_value) for label_value in labels],
        },
        "coordinate_origin": "top-left",
        "coordinate_indexing": "zero-based pixels",
        "bounding_box_format": "x_min, y_min, x_max, y_max (inclusive)",
    }
    if metadata:
        output_metadata.update(metadata)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    svImage_withInfo(rgb_canvas, output_metadata, output_path)
    LOGGER.info(
        "黒点の外接矩形確認PNGを保存しました: %s (黒点数: %d)",
        output_path,
        len(labels),
    )
    return int(len(labels))


def save_sunspot_exclusion_csv(
    label_image: np.ndarray,
    mask_regions: list[dict[str, Any]],
    output_path: str | Path,
    min_area_exclusions: list[tuple[np.ndarray, list[int], str, int]] | None = None,
    radial_mask_details: dict[str, Any] | None = None,
) -> int:
    """mask、Rmask、minArea で除外された黒点を、要因別に CSV へ保存する。"""
    if label_image.ndim != 2:
        raise ValueError("黒点ラベル画像は2次元配列にしてください。")

    mask_exclusions_by_label: dict[int, list[dict[str, int]]] = {}
    for region in mask_regions:
        mask_id = int(region["index"])
        mask_info = {
            "mask_id": mask_id,
            "x": int(region["x"]),
            "y": int(region["y"]),
            "width": int(region["width"]),
            "height": int(region["height"]),
        }
        excluded_labels = region.get(
            "excluded_candidate_labels",
            region.get("overlapping_candidate_labels", []),
        )
        for label in excluded_labels:
            mask_exclusions_by_label.setdefault(int(label), []).append(mask_info)

    radial_exclusions_by_label: dict[int, dict[str, float]] = {}
    if radial_mask_details and radial_mask_details.get("enabled", False):
        for candidate in radial_mask_details.get("excluded_candidates", []):
            radial_exclusions_by_label[int(candidate["label"])] = {
                "centroid_distance_px": float(candidate["centroid_distance_px"]),
            }

    mask_and_radial_labels = sorted(
        set(mask_exclusions_by_label) | set(radial_exclusions_by_label)
    )

    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "sunspot_id",
        "label",
        "centroid_x_px",
        "centroid_y_px",
        "bbox_x_px",
        "bbox_y_px",
        "bbox_width_px",
        "bbox_height_px",
        "area_px",
        "excluded_by_factor",
        "exclusion_stage",
        "min_area_threshold_px",
        "excluded_by_mask_ids",
        "excluded_by_mask_regions",
        "radial_distance_px",
        "radial_limit_px",
        "radial_ratio",
        "radial_center_x_px",
        "radial_center_y_px",
    ]
    rows: list[dict[str, Any]] = []

    def append_exclusions(
        source_labels: np.ndarray,
        excluded_labels: list[int],
        factor: str,
        stage: str,
        min_area_threshold: int | None = None,
        masks_by_label: dict[int, list[dict[str, int]]] | None = None,
        factors_by_label: dict[int, list[str]] | None = None,
        radial_by_label: dict[int, dict[str, float]] | None = None,
    ) -> None:
        """指定されたラベル画像から除外候補の統計行を追加する。"""
        if source_labels.ndim != 2:
            raise ValueError("除外候補のラベル画像は2次元配列にしてください。")
        for spot_label in sorted(set(excluded_labels)):
            y_coordinates, x_coordinates = np.nonzero(source_labels == spot_label)
            if x_coordinates.size == 0:
                LOGGER.warning(
                    "除外一覧にある黒点ラベル %d が元のラベル画像に存在しません。"
                    " factor=%s, stage=%s",
                    spot_label,
                    factor,
                    stage,
                )
                continue

            min_x = int(x_coordinates.min())
            max_x = int(x_coordinates.max())
            min_y = int(y_coordinates.min())
            max_y = int(y_coordinates.max())
            excluded_by_masks = (masks_by_label or {}).get(spot_label, [])
            mask_ids = [mask["mask_id"] for mask in excluded_by_masks]
            candidate_factors = (factors_by_label or {}).get(spot_label, [factor])
            radial_info = (radial_by_label or {}).get(spot_label, {})
            has_radial_factor = "Rmask" in candidate_factors
            rows.append(
                {
                    "label": int(spot_label),
                    "centroid_x_px": round(float(x_coordinates.mean()), 6),
                    "centroid_y_px": round(float(y_coordinates.mean()), 6),
                    "bbox_x_px": min_x,
                    "bbox_y_px": min_y,
                    "bbox_width_px": max_x - min_x + 1,
                    "bbox_height_px": max_y - min_y + 1,
                    "area_px": int(x_coordinates.size),
                    "excluded_by_factor": ";".join(candidate_factors),
                    "exclusion_stage": stage,
                    "min_area_threshold_px": (
                        min_area_threshold if min_area_threshold is not None else ""
                    ),
                    "excluded_by_mask_ids": ";".join(
                        str(mask_id) for mask_id in mask_ids
                    ),
                    "excluded_by_mask_regions": (
                        json.dumps(excluded_by_masks, ensure_ascii=False)
                        if excluded_by_masks
                        else ""
                    ),
                    "radial_distance_px": (
                        round(radial_info["centroid_distance_px"], 6)
                        if has_radial_factor
                        else ""
                    ),
                    "radial_limit_px": (
                        round(float(radial_mask_details["threshold_radius_px"]), 6)
                        if has_radial_factor and radial_mask_details
                        else ""
                    ),
                    "radial_ratio": (
                        float(radial_mask_details["radius_ratio"])
                        if has_radial_factor and radial_mask_details
                        else ""
                    ),
                    "radial_center_x_px": (
                        float(radial_mask_details["center_x_px"])
                        if has_radial_factor and radial_mask_details
                        else ""
                    ),
                    "radial_center_y_px": (
                        float(radial_mask_details["center_y_px"])
                        if has_radial_factor and radial_mask_details
                        else ""
                    ),
                }
            )

    factors_by_label: dict[int, list[str]] = {}
    for label in sorted(mask_exclusions_by_label):
        factors_by_label.setdefault(label, []).append("mask")
    for label in sorted(radial_exclusions_by_label):
        factors_by_label.setdefault(label, []).append("Rmask")

    if mask_and_radial_labels:
        append_exclusions(
            label_image,
            mask_and_radial_labels,
            "mask",
            "mask",
            masks_by_label=mask_exclusions_by_label,
            factors_by_label=factors_by_label,
            radial_by_label=radial_exclusions_by_label,
        )

    for source_labels, excluded_labels, stage, min_area_threshold in (
        min_area_exclusions or []
    ):
        append_exclusions(
            source_labels,
            excluded_labels,
            "minArea",
            stage,
            min_area_threshold,
        )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for sunspot_id, row in enumerate(rows, start=1):
            writer.writerow({"sunspot_id": sunspot_id, **row})

    mask_excluded_count = len(mask_exclusions_by_label)
    radial_excluded_count = len(radial_exclusions_by_label)
    min_area_excluded_count = sum(
        len(excluded_labels)
        for _source_labels, excluded_labels, _stage, _min_area in (
            min_area_exclusions or []
        )
    )
    LOGGER.info(
        "黒点除外一覧 CSV を保存しました: %s (合計: %d、mask: %d、Rmask: %d、minArea: %d)",
        csv_path,
        len(rows),
        mask_excluded_count,
        radial_excluded_count,
        min_area_excluded_count,
    )
    for label, masks in mask_exclusions_by_label.items():
        LOGGER.debug(
            "除外黒点 label=%d: factor=mask, mask_ids=%s",
            label,
            [mask["mask_id"] for mask in masks],
        )
    for label, radial_info in radial_exclusions_by_label.items():
        LOGGER.debug(
            "除外黒点 label=%d: factor=Rmask, centroid_distance_px=%.3f",
            label,
            radial_info["centroid_distance_px"],
        )
    return len(rows)

def closingCircle(image: np.ndarray, kernel: np.ndarray | int) -> np.ndarray:
    """
    楕円形またはカスタムカーネルを使用して、画像に対するモルフォロジー閉じ処理(Closing)を実行します。

    Parameters
    ----------
    image : np.ndarray
        処理対象の入力画像。
    kernel : np.ndarray または int
        楕円形構造要素のサイズ(整数の場合)、またはカスタム構造要素の行列(np.ndarrayの場合)。

    Returns
    -------
    np.ndarray
        モルフォロジー閉じ処理が適用された画像。

    Raises
    ------
    TypeError
        引数`kernel` が整数でも numpy 配列でもない場合。
    """
    if isinstance(kernel, int):
        adopt_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    elif isinstance(kernel, np.ndarray):
        adopt_kernel = kernel
    else:
        raise TypeError("Invalid kernel type. Expected int or np.ndarray.")

    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, adopt_kernel)

    return closed


available_fonts = {font.name for font in font_manager.fontManager.ttflist}


def save_processing_summary(
    images: list[tuple[str, np.ndarray]],
    config: dict,
    image_path: Path,
    count: int,
    output_path: Path,
) -> None:
    """処理段階ごとの画像と実行情報をまとめたレポートを保存する。

    Parameters
    ----------
    images : list[tuple[str, numpy.ndarray]]
        表示する画像と各画像のタイトル。
    config : dict
        レポートに記載する実行設定。
    image_path : pathlib.Path
        入力画像のパス。
    count : int
        検出した黒点候補数。
    output_path : pathlib.Path
        レポート画像の保存先。

    Returns
    -------
    None
    """
    # 日本語フォントを追加
    for font_name in ("Yu Gothic", "Meiryo", "MS Gothic"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.8, 2.2, 2.2],
        hspace=0.35,
        wspace=0.15,
    )

    # 上部：基本情報
    ax_info = fig.add_subplot(grid[0, :])
    ax_info.axis("off")
    ax_info.text(
        0.01,
        0.95,
        "Solar Image Processing Summary",
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    ax_info.text(
        0.01,
        0.62,
        f"Input file: {image_path.name}\nGenerated: {generated_at}\nConnected components: {count}",
        fontsize=11,
        va="top",
    )

    # Paths を除外した設定値を表示する
    display_config = {
        key: value
        for key, value in config.items()
        if key != "Paths"
    }

    ax_info.text(
        0.48,
        0.95,
        "設定値:\n" + pformat(display_config, sort_dicts=False),
        fontsize=8,
        va="top",
    )

    # 下部：4段階の画像
    for i, (title, image) in enumerate(images):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        # 二値画像は0と255をそのまま白黒表示する
        if i == 3:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image, cmap="gray")

    fig.subplots_adjust(top=0.96, bottom=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def output_image(
    raw_image: np.ndarray,
    limb_corrected_image: np.ndarray,
    emphasized_image: np.ndarray,
    binary_image: np.ndarray,
    config: dict[str, Any],
    image_path: Path | str,
    count: int,
    output_dir: Path | str | None = None,
) -> None:
    """処理段階ごとの画像と実行情報を1枚の画像に保存する。

    Parameters
    ----------
    raw_image : numpy.ndarray
        入力画像。
    limb_corrected_image : numpy.ndarray
        周縁減光補正後の画像。
    emphasized_image : numpy.ndarray
        特徴強調後の画像。
    binary_image : numpy.ndarray
        二値化後の画像。
    config : dict[str, Any]
        処理に使用した設定。
    image_path : pathlib.Path | str
        入力画像のパス。
    count : int
        検出した黒点候補数。
    output_dir : pathlib.Path | str | None, optional
        サマリー画像の保存先ディレクトリ。未指定の場合は従来どおり output/ を使用する。

    Returns
    -------
    None
    """
    images = [
        ("1. Raw Image", raw_image),
        ("2. Limb Darkening Corrected", limb_corrected_image),
        ("3. Enhanced Image", emphasized_image),
        ("4. Binary Image", binary_image),
    ]

    summary_directory = Path(output_dir) if output_dir is not None else Path("output")
    output_path = summary_directory / f"{Path(image_path).stem}_summary.png"

    save_processing_summary(
        images=images,
        config=config,
        image_path=Path(image_path),
        count=count,
        output_path=output_path,
    )

    LOGGER.info("処理結果レポートを保存しました: %s", output_path)
