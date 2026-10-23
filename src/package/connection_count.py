
"""Count connected components in a thresholded solar image."""

import numpy as np
from scipy import ndimage


def count_connected_components(
    image: np.ndarray,
    c: float = 1,
    space_threshold: float = 1000,
) -> tuple[int, np.ndarray]:
    """太陽像を二値化し、黒点候補の連結成分数を返す。

    Args:
        image: 二値化前の2次元太陽画像を表すNumPy配列。
        c: 閾値を決める係数。μ-cσ（μ:平均値、σ:標準偏差）以下の画素を黒点とする。デフォルトは1。
        space_threshold: 宇宙空間を判定する閾値。この値以下の画素を宇宙空間とする。デフォルトは1000。

    Returns
    -------
        count: 黒点候補の連結成分数。
        binary_image: 黒点候補を黒（0）、その他を白（255）とした画像。
    """
    # 太陽像の領域を抽出
    solar_mask = image > space_threshold

    # 太陽像の画素だけで平均値と標準偏差を計算
    solar_pixels = image[solar_mask]
    if solar_pixels.size == 0:
        return 0, np.full(image.shape, 255, dtype=np.uint8)

    mean = np.mean(solar_pixels)
    std = np.std(solar_pixels)

    # 閾値を計算
    threshold = mean - c * std

    # 黒点候補を黒（0）、その他を白（255）にする
    sunspot_mask = solar_mask & (image < threshold)
    binary_image = np.full(image.shape, 255, dtype=np.uint8)
    binary_image[sunspot_mask] = 0

    # 連結成分を数える
    _, count = ndimage.label(sunspot_mask)  # pyright: ignore[reportGeneralTypeIssues]

    return count, binary_image