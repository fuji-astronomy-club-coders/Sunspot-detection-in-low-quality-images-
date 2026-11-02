
"""Count connected components in a thresholded solar image."""

import numpy as np
from scipy import ndimage

version = "1.00"

def countConnectedComponents(
    image: np.ndarray,
    c: float = 1,
    space_threshold: float = 1000,
    min_area: int = 5,
) -> tuple[int, np.ndarray, np.ndarray]:
    """太陽像を二値化し、小領域を除去して黒点候補の連結成分数を返す。

    Parameters
    ----------
    image : numpy.ndarray
        二値化前の2次元太陽画像。
    c : float, optional
        閾値を決める係数。平均値と標準偏差から候補を抽出する。
    space_threshold : float, optional
        太陽像と宇宙空間を区別する画素値のしきい値。
    min_area : int, optional
        黒点候補として残す連結成分の最小画素数。

    Returns
    -------
    tuple[int, numpy.ndarray, numpy.ndarray]
        連結成分数、黒点候補を黒（0）・その他を白（255）とした画像、ラベル画像。
    """
    # 太陽像の領域を抽出
    solar_mask = image > space_threshold

    # 太陽像の画素だけで平均値と標準偏差を計算
    solar_pixels = image[solar_mask]
    if solar_pixels.size == 0:
        empty_image = np.full(image.shape, 255, dtype=np.uint8)
        empty_labels = np.zeros(image.shape, dtype=np.intp)
        return 0, empty_image, empty_labels

    mean = np.mean(solar_pixels)
    std = np.std(solar_pixels)

    # 閾値を計算
    threshold = mean + c * std

    # 黒点候補を抽出
    sunspot_mask = solar_mask & (image > threshold)

    # 連結成分をラベリング
    labeled_image, count = ndimage.label(sunspot_mask) # pyright: ignore[reportGeneralTypeIssues]
    labeled_image = labeled_image.astype(np.intp)

    # 各連結成分の面積を計算
    component_sizes = np.bincount(labeled_image.ravel())

    # 最小面積以上の連結成分だけ残す
    keep_mask = component_sizes >= min_area
    keep_mask[0] = False

    filtered_mask = keep_mask[labeled_image]

    # 小領域除去後のラベル画像を作成
    filtered_labels = np.where(filtered_mask, labeled_image, 0)

    # 二値画像を作成
    binary_image = np.full(image.shape, 255, dtype=np.uint8)
    binary_image[filtered_mask] = 0

    # 検出結果を表示
    print("太陽領域の画素数:", np.count_nonzero(solar_mask))
    print("閾値:", threshold)
    print("黒点候補の画素数:", np.count_nonzero(filtered_mask))
    print("黒点候補の連結成分数:", np.count_nonzero(keep_mask) - 1)

    count = int(np.count_nonzero(keep_mask) - 1)

    return count, binary_image, filtered_labels
