"""プロジェクト全体で使用する共通ユーティリティを提供するモジュール。

複数のモジュールから再利用される汎用的な処理を集約する。
特定の機能や実行モードに依存する処理は ``src/package/`` 配下に配置し、
本モジュールには共通性の高い処理のみを実装する。

アプリケーションのメイン処理は ``src/main.py`` に、
実行制御はプロジェクトルートの ``run.py`` に配置する。
"""
import base64
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, cast
from xml.sax.saxutils import escape as xml_escape

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


def save_sunspot_bounding_box_files(
    reference_image: np.ndarray,
    label_image: np.ndarray,
    output_path: str | Path,
    options: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Save a source-image-backed PNG and vector SVG with sunspot annotations.

    The emphasized source image is the default background. Labels and boxes
    remain separate annotations, and SVG annotation elements stay vector
    while the background is embedded as a raster PNG.

    Set background_mode to transparent to export annotations without an image
    background.

    Parameters
    ----------
    reference_image : numpy.ndarray
        Two-dimensional emphasized image used as the background when
        background_mode is image.
    label_image : numpy.ndarray
        Two-dimensional integer label image after mask exclusions.
    output_path : str | pathlib.Path
        PNG output path. The SVG uses the same path with a .svg suffix.
    options : dict[str, Any] | None, optional
        Options from the BoundingBox section of config.yaml.
    metadata : dict[str, Any] | None, optional
        Additional metadata stored in the PNG and SVG.

    Returns
    -------
    int
        Number of labeled components represented in the overlay.
    """
    if reference_image.ndim != 2 or label_image.ndim != 2:
        raise ValueError("参照画像と黒点ラベル画像は2次元配列にしてください。")
    if reference_image.shape != label_image.shape:
        raise ValueError("参照画像と黒点ラベル画像のサイズが一致していません。")

    settings = options or {}
    background_mode = settings.get("background_mode", "image")
    if background_mode not in ("image", "transparent"):
        raise ValueError(
            "BoundingBox.background_mode は image / transparent のいずれかにしてください。"
        )

    def read_color(name: str, default: list[int]) -> tuple[int, int, int]:
        value = settings.get(name, default)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(
                isinstance(channel, bool)
                or not isinstance(channel, (int, np.integer))
                or not 0 <= int(channel) <= 255
                for channel in value
            )
        ):
            raise ValueError(f"BoundingBox.{name} は0〜255のRGB値3個で指定してください。")
        return (int(value[0]), int(value[1]), int(value[2]))

    def read_positive_int(name: str, default: int) -> int:
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"BoundingBox.{name} は正の整数にしてください。")
        return int(value)

    def read_nonnegative_int(name: str, default: int) -> int:
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"BoundingBox.{name} は0以上の整数にしてください。")
        return int(value)

    def read_bool(name: str, default: bool) -> bool:
        value = settings.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"BoundingBox.{name} は true / false で指定してください。")
        return value

    box_color = read_color("box_color", [0, 255, 0])
    label_color = read_color("label_color", [255, 255, 0])
    label_outline_color = read_color("label_outline_color", [0, 0, 0])
    leader_line_color = read_color("leader_line_color", [255, 255, 255])
    line_width = read_positive_int("line_width", 2)
    label_thickness = read_positive_int("label_thickness", 1)
    label_outline_thickness = read_positive_int("label_outline_thickness", 3)
    label_padding = read_nonnegative_int("label_padding", 3)
    label_gap = read_nonnegative_int("label_gap", 5)
    save_png = read_bool("save_png", True)
    save_svg = read_bool("save_svg", True)
    collision_avoidance = read_bool("collision_avoidance", True)
    show_leader_lines = read_bool("show_leader_lines", False)
    font_scale_value = settings.get("label_font_scale", 0.6)
    if (
        isinstance(font_scale_value, bool)
        or not isinstance(font_scale_value, (int, float))
        or not math.isfinite(float(font_scale_value))
        or float(font_scale_value) <= 0
    ):
        raise ValueError("BoundingBox.label_font_scale は正の有限値にしてください。")
    font_scale = float(font_scale_value)

    height, width = label_image.shape
    labels = np.unique(label_image)
    labels = labels[labels > 0]
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    def intersects(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        return (
            first[0] < second[2]
            and second[0] < first[2]
            and first[1] < second[3]
            and second[1] < first[3]
        )

    components: list[dict[str, Any]] = []
    for label_value in labels:
        y_coordinates, x_coordinates = np.nonzero(label_image == label_value)
        if x_coordinates.size == 0:
            continue
        x1 = int(x_coordinates.min())
        x2 = int(x_coordinates.max())
        y1 = int(y_coordinates.min())
        y2 = int(y_coordinates.max())
        label_text = str(int(label_value))
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            font_face,
            font_scale,
            label_thickness,
        )
        label_width = text_width + 2 * label_padding
        label_height = text_height + baseline + 2 * label_padding
        components.append(
            {
                "label": int(label_value),
                "text": label_text,
                "bbox": (x1, y1, x2 + 1, y2 + 1),
                "bbox_inclusive": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "text_width": text_width,
                "text_height": text_height,
                "baseline": baseline,
                "label_width": label_width,
                "label_height": label_height,
            }
        )

    # Larger components get first choice of nearby label positions.
    components.sort(
        key=lambda item: (
            -(
                (item["bbox"][2] - item["bbox"][0])
                * (item["bbox"][3] - item["bbox"][1])
            ),
            item["label"],
        )
    )
    bounding_rectangles = [item["bbox"] for item in components]
    occupied_label_rectangles: list[tuple[int, int, int, int]] = []
    label_layout: list[dict[str, int]] = []
    max_dimension = max(width, height)
    for component in components:
        x1, y1, x2, y2 = component["bbox"]
        center_x, center_y = component["center"]
        label_width = component["label_width"]
        label_height = component["label_height"]
        step_x = max(1, label_width + label_gap)
        step_y = max(1, label_height + label_gap)
        preferred_positions = [
            (round(center_x - label_width / 2), y1 - label_height - label_gap),
            (round(center_x - label_width / 2), y2 + label_gap),
            (x2 + label_gap, round(center_y - label_height / 2)),
            (x1 - label_width - label_gap, round(center_y - label_height / 2)),
        ]
        max_rings = min(
            64,
            max(4, math.ceil(max_dimension / max(1, min(step_x, step_y)))),
        )
        chosen_rectangle: tuple[int, int, int, int] | None = None

        def try_layout(
            *,
            avoid_bounding_boxes: bool,
            max_rings: int = max_rings,
            preferred_positions: list[tuple[int, int]] = preferred_positions,
            step_x: int = step_x,
            step_y: int = step_y,
            label_width: int = label_width,
            label_height: int = label_height,
        ) -> tuple[int, int, int, int] | None:
            for ring in range(max_rings + 1):
                offsets = [
                    (dx, dy)
                    for dx in range(-ring, ring + 1)
                    for dy in range(-ring, ring + 1)
                    if max(abs(dx), abs(dy)) == ring
                ]
                offsets.sort(key=lambda offset: (offset[0] ** 2 + offset[1] ** 2, offset[1], offset[0]))
                for base_x, base_y in preferred_positions:
                    for dx, dy in offsets:
                        left = int(base_x + dx * step_x)
                        top = int(base_y + dy * step_y)
                        candidate = (left, top, left + label_width, top + label_height)
                        if (
                            left < 0
                            or top < 0
                            or candidate[2] > width
                            or candidate[3] > height
                        ):
                            continue
                        if collision_avoidance and any(
                            intersects(candidate, occupied)
                            for occupied in occupied_label_rectangles
                        ):
                            continue
                        if avoid_bounding_boxes and collision_avoidance and any(
                            intersects(candidate, box)
                            for box in bounding_rectangles
                        ):
                            continue
                        return candidate
            return None

        chosen_rectangle = try_layout(avoid_bounding_boxes=True)
        if chosen_rectangle is None:
            # In a tightly packed group, allow label padding to cross a box
            # edge, but never allow two label backgrounds/glyph areas to overlap.
            chosen_rectangle = try_layout(avoid_bounding_boxes=False)
        if chosen_rectangle is None:
            LOGGER.warning(
                "黒点ラベル %d の重なりを完全には解消できませんでした。",
                component["label"],
            )
            left = min(max(0, preferred_positions[0][0]), max(0, width - label_width))
            top = min(max(0, preferred_positions[0][1]), max(0, height - label_height))
            chosen_rectangle = (left, top, left + label_width, top + label_height)

        occupied_label_rectangles.append(chosen_rectangle)
        component["label_rectangle"] = chosen_rectangle
        label_layout.append(
            {
                "label": component["label"],
                "x": chosen_rectangle[0],
                "y": chosen_rectangle[1],
                "width": label_width,
                "height": label_height,
                "bbox_x1": component["bbox_inclusive"][0],
                "bbox_y1": component["bbox_inclusive"][1],
                "bbox_x2": component["bbox_inclusive"][2],
                "bbox_y2": component["bbox_inclusive"][3],
            }
        )

    output_metadata: dict[str, Any] = {
        "label_mapping": {
            "csv_file": "sunspot_statistics.csv",
            "csv_column": "label",
            "labels": sorted(int(item["label"]) for item in components),
        },
        "label_layout": json.dumps(label_layout, ensure_ascii=False),
        "coordinate_origin": "top-left",
        "coordinate_indexing": "zero-based pixels",
        "bounding_box_format": "x_min, y_min, x_max, y_max (inclusive)",
        "background": background_mode,
        "vector_format": "SVG",
    }
    if metadata:
        output_metadata.update(metadata)

    box_rgb: tuple[int, int, int] = box_color
    label_rgb: tuple[int, int, int] = label_color
    outline_rgb: tuple[int, int, int] = label_outline_color
    leader_rgb: tuple[int, int, int] = leader_line_color

    # Normalize the emphasized image for an 8-bit display background. The
    # source pixels and extraction labels remain unchanged.
    background_gray = np.zeros((height, width), dtype=np.uint8)
    if background_mode == "image":
        source = np.asarray(reference_image, dtype=np.float64)
        finite_mask = np.isfinite(source)
        if np.any(finite_mask):
            finite_values = source[finite_mask]
            low_percentile = settings.get("background_low_percentile", 1.0)
            high_percentile = settings.get("background_high_percentile", 99.5)
            if (
                isinstance(low_percentile, bool)
                or isinstance(high_percentile, bool)
                or not isinstance(low_percentile, (int, float))
                or not isinstance(high_percentile, (int, float))
                or not 0 <= float(low_percentile) < float(high_percentile) <= 100
            ):
                raise ValueError(
                    "BoundingBox.background_low_percentile / "
                    "background_high_percentile は 0 <= low < high <= 100 にしてください。"
                )
            low_value, high_value = np.percentile(
                finite_values,
                [float(low_percentile), float(high_percentile)],
            )
            if high_value > low_value:
                normalized = np.zeros((height, width), dtype=np.float64)
                normalized[finite_mask] = (
                    (finite_values - low_value) * 255.0 / (high_value - low_value)
                )
                background_gray = np.clip(normalized, 0.0, 255.0).astype(np.uint8)

    if background_mode == "image":
        drawing_canvas = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
        box_cv = (box_rgb[2], box_rgb[1], box_rgb[0])
        label_cv = (label_rgb[2], label_rgb[1], label_rgb[0])
        outline_cv = (outline_rgb[2], outline_rgb[1], outline_rgb[0])
        leader_cv = (leader_rgb[2], leader_rgb[1], leader_rgb[0])
    else:
        drawing_canvas = np.zeros((height, width, 4), dtype=np.uint8)
        box_cv = (*box_rgb[::-1], 255)
        label_cv = (*label_rgb[::-1], 255)
        outline_cv = (*outline_rgb[::-1], 255)
        leader_cv = (*leader_rgb[::-1], 255)

    if save_png:
        for component in components:
            x1, y1, x2, y2 = component["bbox_inclusive"]
            cv2.rectangle(
                drawing_canvas,
                (x1, y1),
                (x2, y2),
                box_cv,
                line_width,
                lineType=cv2.LINE_AA,
            )
        if show_leader_lines:
            for component in components:
                left, top, right, bottom = component["label_rectangle"]
                center_x, center_y = component["center"]
                target_x = min(max(center_x, left), right)
                target_y = min(max(center_y, top), bottom)
                cv2.line(
                    drawing_canvas,
                    (int(round(center_x)), int(round(center_y))),
                    (int(round(target_x)), int(round(target_y))),
                    leader_cv,
                    1,
                    lineType=cv2.LINE_AA,
                )
        for component in components:
            left, top, _right, _bottom = component["label_rectangle"]
            text_x = left + label_padding
            text_baseline = top + label_padding + component["text_height"]
            cv2.putText(
                drawing_canvas,
                component["text"],
                (text_x, text_baseline),
                font_face,
                font_scale,
                outline_cv,
                label_thickness + label_outline_thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                drawing_canvas,
                component["text"],
                (text_x, text_baseline),
                font_face,
                font_scale,
                label_cv,
                label_thickness,
                lineType=cv2.LINE_AA,
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    if save_png:
        png_metadata = dict(output_metadata)
        png_metadata["config"] = settings
        if background_mode == "image":
            png_image = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2RGB)
        else:
            png_image = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGRA2RGBA)
        svImage_withInfo(png_image, png_metadata, output_path)
        saved_files.append(str(output_path))

    if save_svg:
        def rgb_hex(color: tuple[int, int, int]) -> str:
            return "#{:02X}{:02X}{:02X}".format(*color)

        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            ),
            f"<metadata>{xml_escape(json.dumps(output_metadata, ensure_ascii=False))}</metadata>",
        ]
        if background_mode == "image":
            success, encoded_background = cv2.imencode(".png", background_gray)
            if not success:
                raise RuntimeError("SVG背景画像のPNGエンコードに失敗しました。")
            background_data = base64.b64encode(encoded_background.tobytes()).decode("ascii")
            svg_parts.append(
                f'<image x="0" y="0" width="{width}" height="{height}" '
                f'href="data:image/png;base64,{background_data}" '
                'preserveAspectRatio="none"/>'
            )
        svg_parts.append(
            '<g fill="none" stroke-linecap="square" stroke-linejoin="round">'
        )
        for component in components:
            x1, y1, x2, y2 = component["bbox_inclusive"]
            svg_parts.append(
                f'<rect x="{x1}" y="{y1}" width="{x2 - x1 + 1}" '
                f'height="{y2 - y1 + 1}" stroke="{rgb_hex(box_rgb)}" '
                f'stroke-width="{line_width}"/>'
            )
        if show_leader_lines:
            for component in components:
                left, top, right, bottom = component["label_rectangle"]
                center_x, center_y = component["center"]
                target_x = min(max(center_x, left), right)
                target_y = min(max(center_y, top), bottom)
                svg_parts.append(
                    f'<line x1="{center_x:.2f}" y1="{center_y:.2f}" '
                    f'x2="{target_x:.2f}" y2="{target_y:.2f}" '
                    f'stroke="{rgb_hex(leader_rgb)}" stroke-width="1"/>'
                )
        svg_parts.append("</g>")
        svg_font_size = max(6.0, font_scale * 22.0)
        for component in components:
            left, top, _right, _bottom = component["label_rectangle"]
            text_x = left + label_padding
            text_baseline = top + label_padding + component["text_height"]
            svg_parts.append(
                f'<text x="{text_x}" y="{text_baseline}" '
                f'font-family="DejaVu Sans, Arial, sans-serif" '
                f'font-size="{svg_font_size:.2f}" font-weight="bold" '
                f'fill="{rgb_hex(label_rgb)}" stroke="{rgb_hex(outline_rgb)}" '
                f'stroke-width="{max(1, label_outline_thickness)}" '
                f'stroke-linejoin="round" paint-order="stroke fill">'
                f'{xml_escape(component["text"])}</text>'
            )
        svg_parts.append("</svg>")
        svg_path = output_path.with_suffix(".svg")
        svg_path.write_text("\n".join(svg_parts) + "\n", encoding="utf-8")
        saved_files.append(str(svg_path))

    if not save_png and not save_svg:
        LOGGER.info("BoundingBox の save_png/save_svg が両方 false のためファイルを保存しません。")
    else:
        LOGGER.info(
            "黒点の外接矩形オーバーレイを保存しました: %s (黒点数: %d)",
            ", ".join(saved_files),
            len(components),
        )
    return len(components)


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
