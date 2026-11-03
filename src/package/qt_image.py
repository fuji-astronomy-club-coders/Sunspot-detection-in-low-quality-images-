"""輝度画像を Qt ダイアログ向けに正規化・着色するユーティリティ。"""

import cv2
import numpy as np

COLORMAPS = {
    "AUTUMN": cv2.COLORMAP_AUTUMN,
    "BONE": cv2.COLORMAP_BONE,
    "JET": cv2.COLORMAP_JET,
    "WINTER": cv2.COLORMAP_WINTER,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "OCEAN": cv2.COLORMAP_OCEAN,
    "SUMMER": cv2.COLORMAP_SUMMER,
    "SPRING": cv2.COLORMAP_SPRING,
    "COOL": cv2.COLORMAP_COOL,
    "HSV": cv2.COLORMAP_HSV,
    "PINK": cv2.COLORMAP_PINK,
    "HOT": cv2.COLORMAP_HOT,
    "PARULA": cv2.COLORMAP_PARULA,
    "TURBO": cv2.COLORMAP_TURBO,
    "TWILIGHT": cv2.COLORMAP_TWILIGHT,
    "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "CIVIDIS": cv2.COLORMAP_CIVIDIS,
}


def normalize_intensity(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """有限な輝度値を表示用の8 bit階調へ正規化する。

    Parameters
    ----------
    image : numpy.ndarray
        2次元の輝度画像。

    Returns
    -------
    tuple[numpy.ndarray, float, float]
        8 bit正規化画像、有限画素の最小値、最大値。

    Raises
    ------
    ValueError
        入力が空または2次元ではない場合。
    """
    if image.ndim != 2 or image.size == 0:
        raise ValueError("表示画像は空でない2次元配列にしてください。")

    values = np.asarray(image, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.zeros(image.shape, dtype=np.uint8), 0.0, 0.0

    finite_values = values[finite_mask]
    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    normalized = np.zeros(image.shape, dtype=np.uint8)

    if maximum > minimum:
        scaled = (values[finite_mask] - minimum) / (maximum - minimum)
        normalized[finite_mask] = np.rint(
            np.clip(scaled, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    else:
        normalized[finite_mask] = 127

    return normalized, minimum, maximum


def colorize_intensity(
    image: np.ndarray,
    colormap: str = "GRAY",
) -> tuple[np.ndarray, float, float]:
    """輝度画像を指定カラーマップの8 bit BGR画像へ変換する。

    Parameters
    ----------
    image : numpy.ndarray
        2次元の輝度画像。
    colormap : str, optional
        OpenCVカラーマップ名。GRAYはグレースケール表示。

    Returns
    -------
    tuple[numpy.ndarray, float, float]
        BGR表示画像、元画像の最小輝度、最大輝度。

    Raises
    ------
    ValueError
        未対応のカラーマップが指定された場合。
    """
    normalized, minimum, maximum = normalize_intensity(image)
    name = colormap.strip().upper()
    if name == "GRAY":
        display_image = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    elif name in COLORMAPS:
        display_image = cv2.applyColorMap(normalized, COLORMAPS[name])
    else:
        supported = ", ".join(["GRAY", *COLORMAPS])
        raise ValueError(f"未対応のカラーマップです: {colormap} (対応: {supported})")

    return display_image, minimum, maximum
