
"""二値化済みの太陽画像にWatershedを適用するモジュール。"""

import logging
from typing import Literal, TypedDict, overload

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

LOGGER = logging.getLogger("sunspot.watershed")


class MinAreaFilterDetails(TypedDict):
    """minArea フィルターで除外された watershed 領域の情報。"""

    unfiltered_labels: np.ndarray
    excluded_labels: list[int]
    min_area: int


@overload
def watershedCount(
    binary_image: np.ndarray,
    min_area: int = 5,
    min_distance: int = 5,
    return_details: Literal[False] = False,
) -> tuple[int, np.ndarray, np.ndarray]: ...


@overload
def watershedCount(
    binary_image: np.ndarray,
    min_area: int = 5,
    min_distance: int = 5,
    return_details: Literal[True],
) -> tuple[int, np.ndarray, np.ndarray, MinAreaFilterDetails]: ...


def watershedCount(
    binary_image: np.ndarray,
    min_area: int = 5,
    min_distance: int = 5,
    return_details: bool = False,
) -> (
    tuple[int, np.ndarray, np.ndarray]
    | tuple[int, np.ndarray, np.ndarray, MinAreaFilterDetails]
):
    """Watershedで黒点候補を分割し、個数を数える。

    Parameters
    ----------
    binary_image : numpy.ndarray
        黒点候補が0、背景が255の2次元二値画像。
    min_area : int, optional
        残す領域の最小画素数。
    min_distance : int, optional
        Watershed のマーカー間に確保する最小距離。
    return_details : bool, optional
        True の場合、minArea 適用前のラベル画像と除外ラベルを追加で返す。

    Returns
    -------
    tuple
        領域数、色分け画像、フィルター後ラベル画像。return_details=True の場合は
        minArea 適用前ラベル画像・除外ラベル・最小面積を含む情報も返す。
    """
    # 黒点候補をTrue、背景をFalseにする
    sunspot_mask = binary_image == 0

    # 黒点候補が存在しない場合
    if not np.any(sunspot_mask):
        empty_image = np.full(
            (*binary_image.shape, 3),
            255,
            dtype=np.uint8,
        )
        empty_labels = np.zeros(binary_image.shape, dtype=np.intp)
        if return_details:
            return 0, empty_image, empty_labels, {
                "unfiltered_labels": empty_labels,
                "excluded_labels": [],
                "min_area": int(min_area),
            }
        return 0, empty_image, empty_labels

    # 距離変換：各画素から背景までの距離を求める
    distance_result = ndimage.distance_transform_edt(sunspot_mask)
    distance = np.asarray(distance_result, dtype=np.float64)

    # 距離画像の局所最大値を検出し、マーカー同士の間隔を確保
    coordinates = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=sunspot_mask.astype(np.uint8),
        exclude_border=False,
    )

    # 検出した座標をマーカー画像に変換
    local_max = np.zeros(sunspot_mask.shape, dtype=bool)
    local_max[tuple(coordinates.T)] = True # pyright: ignore


    # 局所最大値にラベルを付ける
    markers, marker_count = ndimage.label(local_max) # pyright: ignore[reportGeneralTypeIssues]

    # マーカーが存在しない場合は元の連結領域を返す
    if marker_count == 0:
        labels, _ = ndimage.label(sunspot_mask) # pyright: ignore[reportGeneralTypeIssues]
    else:
        # 距離画像の高い部分から領域を分割
        watershed_image = -distance
        labels = watershed(
            watershed_image,
            markers=markers,
            mask=sunspot_mask,
        )

    # 各領域の画素数を数える
    labels = np.asarray(labels, dtype=np.intp)
    component_sizes = np.bincount(labels.ravel())
    excluded_labels = [
        int(label)
        for label in np.flatnonzero(component_sizes < min_area)
        if label > 0
    ]

    # 最小面積以上の領域だけ残す
    keep_mask = component_sizes >= min_area
    keep_mask[0] = False

    filtered_mask = keep_mask[labels]

    # 小さな領域を除外したラベル画像を作成
    filtered_labels = np.where(filtered_mask, labels, 0)

     # ラベル画像をカラフルなRGB画像に変換
    normalized_labels = filtered_labels.astype(np.float64)
    max_label = int(normalized_labels.max())

    if max_label > 0:
        normalized_labels /= max_label

    result_image = (
        plt.get_cmap("nipy_spectral")(normalized_labels)[..., :3] * 255
    ).astype(np.uint8)

    # ラベル0は除外済みなので、そのまま数える。
    count = int(np.count_nonzero(keep_mask))
    LOGGER.info("Watershedのマーカー数: %d", marker_count)
    LOGGER.info("Watershed後の領域数: %d", len(np.unique(labels)) - 1)
    LOGGER.info(
        "Watershed minArea フィルター: 最小面積=%d px、除外=%d、残存=%d",
        min_area,
        len(excluded_labels),
        count,
    )

    if return_details:
        return count, result_image, filtered_labels, {
            "unfiltered_labels": labels,
            "excluded_labels": excluded_labels,
            "min_area": int(min_area),
        }
    return count, result_image, filtered_labels
