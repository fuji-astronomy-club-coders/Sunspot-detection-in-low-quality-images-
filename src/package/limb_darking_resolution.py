"""周縁減光を補正するプログラム"""

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
from numpy import dtype, floating, ndarray

# 先輩が作った関数をインポート
from src.module import MIN2ver2 as min2_module
from src.module.MIN2ver2 import MIN2_ignore_sunspots as min2

LOGGER = logging.getLogger("sunspot.limb_darkening")

version = "0.0.2"
min2_version = min2_module.version


class _Min2Kwargs(TypedDict):
    """Keyword arguments accepted by the MIN2 implementation."""

    n: int
    light_threshold: int
    limb_wigth: int | str
    iter_cycles: int
    show: bool
    debug: bool
    show_simple: bool


_MIN2_DEFAULTS: dict[str, int | str | bool] = {
    "n": 10,
    "light_threshold": 50,
    "limb_width": "1/30",
    "iter_cycles": 2,
    "show": False,
    "debug": False,
    "show_simple": False,
}

_MIN2_RESULT_FORMAT = "solar_disk_detection.min2_result"
_MIN2_RESULT_FORMAT_VERSION = 2


def _resolve_min2_settings(
    settings: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """Separate application-only cache settings from MIN2 algorithm parameters."""
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ValueError("MIN2 は YAML mapping にしてください。")
    load_saved_result = settings.get("load_saved_result", False)
    if not isinstance(load_saved_result, bool):
        raise ValueError("MIN2.load_saved_result は true または false にしてください。")
    algorithm_settings = {
        key: value for key, value in settings.items() if key != "load_saved_result"
    }
    return algorithm_settings, load_saved_result


def _load_saved_min2_result(
    image: np.ndarray,
    image_path: str | Path,
) -> tuple[tuple[float, float], float] | None:
    """Load a matching MIN2 JSON sidecar, returning None when it is absent or invalid."""
    source_path = Path(image_path)
    result_path = source_path.with_name(f"{source_path.stem}_min2.json")
    if not result_path.is_file():
        return None

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("結果ファイルのルートはJSON objectである必要があります。")
        if payload.get("format") != _MIN2_RESULT_FORMAT:
            raise ValueError("未対応のMIN2結果ファイル形式です。")
        if payload.get("format_version") != _MIN2_RESULT_FORMAT_VERSION:
            raise ValueError("未対応のMIN2結果ファイルバージョンです。")

        source = payload.get("source_image")
        if not isinstance(source, dict) or source.get("name") != source_path.name:
            raise ValueError("結果ファイルの元画像名が一致しません。")
        if source.get("shape") != [int(image.shape[0]), int(image.shape[1])]:
            raise ValueError("結果ファイルの画像サイズが現在の画像と一致しません。")
        if source.get("size_bytes") != source_path.stat().st_size:
            raise ValueError("結果ファイルの元画像サイズ（byte）が一致しません。")
        saved_hash = source.get("sha256")
        if not isinstance(saved_hash, str) or len(saved_hash) != 64:
            raise ValueError("結果ファイルに有効な元画像SHA-256がありません。")
        image_hash = hashlib.sha256()
        with source_path.open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                image_hash.update(chunk)
        if image_hash.hexdigest() != saved_hash.lower():
            raise ValueError("元画像のSHA-256が一致しません。")

        configuration = payload.get("configuration")
        if (
            not isinstance(configuration, dict)
            or not isinstance(configuration.get("requested"), dict)
            or not isinstance(configuration.get("effective"), dict)
        ):
            raise ValueError("再現に必要なMIN2設定情報がありません。")
        trace = payload.get("detection_trace")
        trace_fields = (
            "sampled_edge_points",
            "initial_retained_points",
            "final_retained_points",
            "excluded_points",
            "iterations",
        )
        if not isinstance(trace, dict) or any(
            not isinstance(trace.get(field), list) for field in trace_fields
        ):
            raise ValueError("再現に必要なMIN2点群・除外履歴がありません。")

        geometry = payload.get("solar_geometry")
        if not isinstance(geometry, dict):
            raise ValueError("太陽円の結果データがありません。")
        values = (
            geometry["center_x_px"],
            geometry["center_y_px"],
            geometry["radius_px"],
        )
        if any(isinstance(value, bool) for value in values):
            raise ValueError("太陽円の座標・半径が数値ではありません。")
        center_x, center_y, radius = (float(value) for value in values)
        if not all(np.isfinite(value) for value in (center_x, center_y, radius)):
            raise ValueError("太陽円の座標・半径に有限値以外が含まれています。")
        if radius <= 0:
            raise ValueError("太陽半径は正の値である必要があります。")
    except (OSError, ValueError, TypeError, KeyError) as error:
        LOGGER.warning(
            "保存済みMIN2結果を利用できません。通常のMIN2計算へ切り替えます: %s (%s)",
            result_path,
            error,
        )
        return None

    LOGGER.info(
        "保存済みMIN2結果を読み込みました: %s (MIN2=%s, 除外点=%d)",
        result_path,
        payload.get("min2_version", "unknown"),
        len(trace["excluded_points"]),
    )
    return (center_x, center_y), radius


def _resolve_min2_kwargs(settings: Mapping[str, Any] | None) -> _Min2Kwargs:
    """Validate MIN2 config and map it to the legacy function API.

    Public configuration uses the correctly spelled limb_width key. The
    original MIN2 function retains its historical limb_wigth parameter name.
    """
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ValueError("MIN2 は YAML mapping にしてください。")

    unknown_keys = sorted(str(key) for key in settings if key not in _MIN2_DEFAULTS)
    if unknown_keys:
        raise ValueError(f"MIN2 に未対応の設定項目があります: {', '.join(unknown_keys)}")

    integer_values: dict[str, int] = {}
    for key in ("n", "light_threshold", "iter_cycles"):
        value = settings.get(key, _MIN2_DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"MIN2.{key} は整数にしてください。")
        minimum = 2 if key == "n" else 0
        if value < minimum:
            raise ValueError(f"MIN2.{key} は {minimum} 以上にしてください。")
        integer_values[key] = value

    limb_width = settings.get("limb_width", _MIN2_DEFAULTS["limb_width"])
    if isinstance(limb_width, bool):
        raise ValueError("MIN2.limb_width は0以上の整数、または分子/分母形式の文字列にしてください。")
    if isinstance(limb_width, int):
        if limb_width < 0:
            raise ValueError("MIN2.limb_width は0以上にしてください。")
    elif isinstance(limb_width, str):
        parts = limb_width.split("/")
        if len(parts) != 2:
            raise ValueError("MIN2.limb_width は \"1/20\" のような分子/分母形式にしてください。")
        try:
            numerator, denominator = (int(part.strip()) for part in parts)
        except ValueError as error:
            raise ValueError("MIN2.limb_width の分子と分母は整数にしてください。") from error
        if numerator < 0 or denominator <= 0:
            raise ValueError("MIN2.limb_width は分子0以上、分母1以上にしてください。")
    else:
        raise ValueError("MIN2.limb_width は0以上の整数、または分子/分母形式の文字列にしてください。")

    boolean_values: dict[str, bool] = {}
    for key in ("show", "debug", "show_simple"):
        value = settings.get(key, _MIN2_DEFAULTS[key])
        if not isinstance(value, bool):
            raise ValueError(f"MIN2.{key} は true または false にしてください。")
        boolean_values[key] = value

    return {
        "n": integer_values["n"],
        "light_threshold": integer_values["light_threshold"],
        "limb_wigth": limb_width,
        "iter_cycles": integer_values["iter_cycles"],
        "show": boolean_values["show"],
        "debug": boolean_values["debug"],
        "show_simple": boolean_values["show_simple"],
    }


def find_solar_geometry(
    image: np.ndarray,
    min2_config: Mapping[str, Any] | None = None,
    *,
    image_path: str | Path | None = None,
) -> tuple[tuple[float, float], float]:
    """MIN2の設定と必要に応じて保存済み結果を使い、太陽中心・半径を取得する。

    MIN2.load_saved_result が有効で、入力画像と同じフォルダに
    <画像名>_min2.json が存在する場合はその結果を再利用します。
    ファイルが無い、または画像名・画像サイズ・内容が一致しない場合はMIN2で再計算します。
    """
    algorithm_settings, load_saved_result = _resolve_min2_settings(min2_config)
    if load_saved_result:
        if image_path is None:
            LOGGER.warning(
                "MIN2.load_saved_result が有効ですが画像パスが指定されていないため、"
                "保存済み結果を検索せずMIN2で計算します。"
            )
        else:
            saved_geometry = _load_saved_min2_result(image, image_path)
            if saved_geometry is not None:
                return saved_geometry

    try:
        (center_x, center_y), radius = min2(
            image, **_resolve_min2_kwargs(algorithm_settings)
        )
    except Exception as error:
        raise ValueError(f"MIN2 による太陽円の推定に失敗しました: {error}") from error

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"MIN2 が不正な太陽半径を返しました: {radius}")

    return (float(center_x), float(center_y)), float(radius)


def polynomialFit(
    image: np.ndarray,
    alpha: float = 0.85,
    brightness_boost: float = 1.1,
    solar_geometry: tuple[tuple[float, float], float] | None = None,
) -> np.ndarray:
    """周縁減光を距離マップと先輩の関数で補正する関数"""
    # 画像の読み込み

    h, w = image.shape

    # 1. 元の画像を使って太陽の中心と半径を取得する
    (cx, cy), max_radius = (
        solar_geometry if solar_geometry is not None else find_solar_geometry(image)
    )

    # 2. 輝度計算用にfloat32へ変換する
    img_float: ndarray[Any, dtype[floating]] = image.astype(np.float32)

    # 2. 画像全体について「中心からの距離（マップ）」を計算する
    Y_indices, X_indices = np.indices((h, w), dtype=np.float32)
    distance_map = np.sqrt((X_indices - cx) ** 2 + (Y_indices - cy) ** 2)

    # 3. 距離ごとの平均輝度（減光プロファイル）を計算する
    mask = distance_map <= max_radius
    max_r_int = int(max_radius)

    profile = np.zeros(max_r_int + 1, dtype=np.float32)
    for r in range(max_r_int + 1):
        ring_mask = (distance_map >= r - 0.5) & (distance_map < r + 0.5) & mask
        if np.any(ring_mask):
            profile[r] = np.mean(img_float[ring_mask])
        else:
            profile[r] = profile[r - 1] if r > 0 else 1.0
        # 0割を防ぐ処理
        profile = np.maximum(profile, 1)

        # プロファイルを滑らかにする
        profile = cv2.GaussianBlur(profile.reshape(-1, 1), (15, 1), 0).flatten()

    # 4. 背景モデルを復元してフラット化する
    distance_clipped = np.clip(distance_map, 0, max_r_int).astype(int)
    background_model = profile[distance_clipped]

    mean_brightness = np.mean(img_float[mask])
    flattened = (img_float / background_model) * mean_brightness

    # 5. 補正の強さ（alpha）と明るさ調整を反映
    corrected = img_float * np.float32(1.0 - alpha) + flattened * np.float32(alpha)
    corrected = corrected * brightness_boost

    # 太陽の外側（黒背景）は元の状態を維持
    corrected[~mask] = img_float[~mask]
    corrected = np.clip(corrected, 0, 2**16 - 1).astype(np.uint16)

    LOGGER.info("太陽の中心: (%.3f, %.3f)", cx, cy)
    LOGGER.info("太陽の半径: %.3f", max_radius)
    LOGGER.info("画像サイズ: %d x %d", w, h)

    return corrected


if __name__ == "__main__":
    input_dir = "."
    image_paths = list(Path(input_dir).glob("img*.tiff"))

    if not image_paths:
        LOGGER.info("対象の画像ファイルが見つかりませんでした")
    else:
        # 保存用のフォルダを自動で作る
        output_dir = Path("corrected_images")
        output_dir.mkdir(exist_ok=True)

        LOGGER.info(f"合計 {len(image_paths)} 枚の画像を処理します...")

        eroCount = 0  # エラー画像のカウントを初期化
        # 1枚ずつ順番に一括処理！
        for path in image_paths:
            filename = Path(path).name
            output_path = output_dir / f"after_{filename}"

            LOGGER.info(f"処理中: {filename} ...")
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                LOGGER.info(f"画像が見つかりません: {path}")
                eroCount += 1
                continue

            corrected_image = polynomialFit(image=img, alpha=0.85, brightness_boost=1.1)
            if not cv2.imwrite(str(output_path), corrected_image):
                LOGGER.info(f"画像の保存に失敗しました: {output_path}")
                eroCount += 1
                continue

        LOGGER.info(
            f"{len(image_paths) - eroCount}/{len(image_paths)}件 の画像の自動処理と保存が成功しました！"
        )
