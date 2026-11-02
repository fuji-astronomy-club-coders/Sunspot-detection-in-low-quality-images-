
"""Count connected components in a thresholded solar image."""

import logging
from typing import Literal, TypedDict, overload

import numpy as np
from scipy import ndimage

LOGGER = logging.getLogger("sunspot.connection_count")


class MinAreaFilterDetails(TypedDict):
    """minArea フィルターで除外された候補の情報。"""

    unfiltered_labels: np.ndarray
    excluded_labels: list[int]
    min_area: int


version = "1.00"


@overload
def countConnectedComponents(
    image: np.ndarray,
    c: float = 1,
    space_threshold: float = 1000,
    min_area: int = 5,
    return_details: Literal[False] = False,
) -> tuple[int, np.ndarray, np.ndarray]: ...


@overload
def countConnectedComponents(
    image: np.ndarray,
    c: float = 1,
    space_threshold: float = 1000,
    min_area: int = 5,
    *,
    return_details: Literal[True],
) -> tuple[int, np.ndarray, np.ndarray, MinAreaFilterDetails]: ...


def countConnectedComponents(
    image: np.ndarray,
    c: float = 1,
    space_threshold: float = 1000,
    min_area: int = 5,
    return_details: bool = False,
) -> (
    tuple[int, np.ndarray, np.ndarray]
    | tuple[int, np.ndarray, np.ndarray, MinAreaFilterDetails]
):
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
    return_details : bool, optional
        True の場合、minArea 適用前のラベル画像と除外ラベルを追加で返す。

    Returns
    -------
    tuple
        連結成分数、二値画像、フィルター後ラベル画像。return_details=True の場合は
        minArea 適用前ラベル画像・除外ラベル・最小面積を含む情報も返す。
    """
    # 太陽像の領域を抽出
    solar_mask = image > space_threshold

    # 太陽像の画素だけで平均値と標準偏差を計算
    solar_pixels = image[solar_mask]
    if solar_pixels.size == 0:
        empty_image = np.full(image.shape, 255, dtype=np.uint8)
        empty_labels = np.zeros(image.shape, dtype=np.intp)
        if return_details:
            return 0, empty_image, empty_labels, {
                "unfiltered_labels": empty_labels,
                "excluded_labels": [],
                "min_area": int(min_area),
            }
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
    excluded_labels = [
        int(label)
        for label in np.flatnonzero(component_sizes < min_area)
        if label > 0
    ]

    # 最小面積以上の連結成分だけ残す
    keep_mask = component_sizes >= min_area
    keep_mask[0] = False

    filtered_mask = keep_mask[labeled_image]

    # 小領域除去後のラベル画像を作成
    filtered_labels = np.where(filtered_mask, labeled_image, 0)

    # 二値画像を作成
    binary_image = np.full(image.shape, 255, dtype=np.uint8)
    binary_image[filtered_mask] = 0

    # 解析時の統計情報はセッションログへ記録する。
    count = int(np.count_nonzero(keep_mask))
    LOGGER.info("太陽領域の画素数: %d", np.count_nonzero(solar_mask))
    LOGGER.info("閾値: %s", threshold)
    LOGGER.info("黒点候補の画素数: %d", np.count_nonzero(filtered_mask))
    LOGGER.info("黒点候補の連結成分数: %d", count)
    LOGGER.info(
        "minArea フィルター: 最小面積=%d px、除外=%d、残存=%d",
        min_area,
        len(excluded_labels),
        count,
    )

    if return_details:
        return count, binary_image, filtered_labels, {
            "unfiltered_labels": labeled_image,
            "excluded_labels": excluded_labels,
            "min_area": int(min_area),
        }
    return count, binary_image, filtered_labels
