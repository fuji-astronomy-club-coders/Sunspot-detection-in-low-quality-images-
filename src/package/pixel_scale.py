"""太陽直径を基準に画像処理パラメーターをpx値へ変換する。"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any


def _parse_fraction(value: str, name: str) -> float:
    """分子/分母形式の割合を検証して浮動小数点数へ変換する。

    Parameters
    ----------
    value : str
        分子/分母形式の割合。
    name : str
        エラー表示に使用する設定名。

    Returns
    -------
    float
        割合。

    Raises
    ------
    ValueError
        分数形式または分子・分母の値が不正な場合。
    """
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError(f'{name} は "1/200" のような分子/分母形式にしてください。')
    try:
        numerator, denominator = (int(part.strip()) for part in parts)
    except ValueError as error:
        raise ValueError(f"{name} の分子と分母は整数にしてください。") from error
    if numerator < 0 or denominator <= 0:
        raise ValueError(f"{name} は分子0以上、分母1以上にしてください。")
    return numerator / denominator


def _validate_diameter(solar_diameter_px: float) -> float:
    """太陽直径が正の有限値であることを確認する。"""
    diameter = float(solar_diameter_px)
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("solar_diameter_px は正の有限値にしてください。")
    return diameter


def resolve_pixel_length(
    value: int | str,
    solar_diameter_px: float,
    name: str,
    *,
    minimum: int = 1,
    odd: bool = False,
) -> int:
    """px値または太陽直径比を、実画像に対応する整数px値へ解決する。

    整数は従来どおり固定px値として扱い、文字列は太陽直径に対する
    分数として解釈する。Sobel等の奇数カーネルには odd=True を指定する。

    Parameters
    ----------
    value : int | str
        固定px値、または分子/分母形式の太陽直径比。
    solar_diameter_px : float
        MIN2が推定した太陽直径（px）。
    name : str
        設定項目名。
    minimum : int, default=1
        解決後に許可する最小px値。
    odd : bool, default=False
        Trueの場合、最も近い正の奇数pxへ丸める。

    Returns
    -------
    int
        解決したpx値。

    Raises
    ------
    ValueError
        設定値・太陽直径・丸め条件が不正な場合。
    """
    diameter = _validate_diameter(solar_diameter_px)
    if isinstance(value, bool):
        raise ValueError(f"{name} は整数px値または分数文字列で指定してください。")
    if isinstance(value, int):
        pixels = value
        if pixels < minimum:
            raise ValueError(f"{name} は {minimum} px 以上にしてください。")
        if odd and pixels % 2 == 0:
            raise ValueError(f"{name} の固定px値には奇数を指定してください。")
        return pixels
    if not isinstance(value, str):
        raise ValueError(f"{name} は整数px値または分数文字列で指定してください。")

    ratio = _parse_fraction(value, name)
    target = diameter * ratio
    if not math.isfinite(target):
        raise ValueError(f"{name} の換算結果が有限値ではありません。")
    if not odd:
        return max(minimum, int(math.floor(target + 0.5)))

    lower = max(1, int(target // 2) * 2 + 1)
    upper = lower + 2
    return min(
        (lower, upper),
        key=lambda candidate: (abs(candidate - target), candidate),
    )


def resolve_pixel_area(
    value: int | str,
    solar_diameter_px: float,
    name: str,
    *,
    minimum: int = 1,
) -> int:
    """px²値または太陽直径比の二乗を、実画像に対応する画素面積へ解決する。

    Parameters
    ----------
    value : int | str
        固定px²値、または太陽直径の二乗に対する分数。
    solar_diameter_px : float
        MIN2が推定した太陽直径（px）。
    name : str
        設定項目名。
    minimum : int, default=1
        解決後に許可する最小画素面積。

    Returns
    -------
    int
        解決した最小画素面積（px²）。

    Raises
    ------
    ValueError
        設定値・太陽直径が不正な場合。
    """
    diameter = _validate_diameter(solar_diameter_px)
    if isinstance(value, bool):
        raise ValueError(f"{name} は整数px²値または分数文字列で指定してください。")
    if isinstance(value, int):
        if value < minimum:
            raise ValueError(f"{name} は {minimum} px² 以上にしてください。")
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} は整数px²値または分数文字列で指定してください。")

    ratio = _parse_fraction(value, name)
    target = diameter**2 * ratio
    if not math.isfinite(target):
        raise ValueError(f"{name} の換算結果が有限値ではありません。")
    return max(minimum, int(math.floor(target + 0.5)))


def resolve_pixel_parameters(
    config: Mapping[str, Any],
    solar_diameter_px: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """config内の空間パラメーターを推定太陽直径に応じたpx値へ解決する。

    Parameters
    ----------
    config : collections.abc.Mapping
        YAMLから読み込んだアプリケーション設定。
    solar_diameter_px : float
        MIN2が推定した太陽直径（px）。

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        実行用にpx値へ変換した設定と、変換結果のメタデータ。
    """
    diameter = _validate_diameter(solar_diameter_px)
    resolved = copy.deepcopy(dict(config))
    effective: dict[str, Any] = {"solar_diameter_px": diameter}

    emphasis = resolved.get("Emphasis", {})
    if isinstance(emphasis, dict):
        for method, key, requires_odd in (
            ("SobelFilter", "kernelSize", True),
            ("BlackTopHat", "kernelSize", True),
            ("GaussianFilter", "sigmaSize", False),
        ):
            settings = emphasis.get(method)
            if isinstance(settings, dict) and key in settings:
                setting_name = f"Emphasis.{method}.{key}"
                value = resolve_pixel_length(
                    settings[key],
                    diameter,
                    setting_name,
                    odd=requires_odd,
                )
                settings[key] = value
                effective[setting_name] = value

    extraction = resolved.get("Extraction", {})
    if isinstance(extraction, dict):
        component_settings = extraction.get("countComponents")
        if isinstance(component_settings, dict) and "min_area" in component_settings:
            setting_name = "Extraction.countComponents.min_area"
            value = resolve_pixel_area(
                component_settings["min_area"],
                diameter,
                setting_name,
            )
            component_settings["min_area"] = value
            effective[setting_name] = value

        watershed_settings = extraction.get("watershed")
        if isinstance(watershed_settings, dict) and "min_distance" in watershed_settings:
            setting_name = "Extraction.watershed.min_distance"
            value = resolve_pixel_length(
                watershed_settings["min_distance"],
                diameter,
                setting_name,
            )
            watershed_settings["min_distance"] = value
            effective[setting_name] = value

    return resolved, effective
