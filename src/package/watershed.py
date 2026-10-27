
"""二値化済みの太陽画像にWatershedを適用するモジュール。"""

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def watershedCount(
    binary_image: np.ndarray,
    min_area: int = 5,
    min_distance: int = 5,
) -> tuple[int, np.ndarray]:
    """Watershedで黒点候補を分割し、個数を数える。

    Args:
        binary_image: 黒点候補が0、背景が255の二値画像。
        min_area: 残す領域の最小画素数。デフォルトは5。
        min_distance: watershedで分割される黒点どうしの最小の距離。

    Return
    -------
        黒点ごとに色分けされた画像。各画素が RGB の3値を持つため、(高さ, 幅, 3) の形状を持ちます。
    """
    # 黒点候補をTrue、背景をFalseにする
    sunspot_mask = binary_image == 0

    # 黒点候補が存在しない場合
    if not np.any(sunspot_mask):
        return 0, np.full(binary_image.shape, 255, dtype=np.uint8)

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

    # 最小面積以上の領域だけ残す
    keep_mask = component_sizes >= min_area
    keep_mask[0] = False

    filtered_mask = keep_mask[labels]

    # 小さな領域を除外したラベル画像を作成
    colored_labels = np.where(filtered_mask, labels, 0)

    # ラベル画像をカラフルなRGB画像に変換
    normalized_labels = colored_labels.astype(np.float64)
    max_label = int(normalized_labels.max())

    if max_label > 0:
        normalized_labels /= max_label

    result_image = (plt.get_cmap("nipy_spectral")(normalized_labels)[..., :3] * 255).astype(np.uint8)

    count = int(np.count_nonzero(keep_mask))

    # ラベル0は除外済みなので、そのまま数える
    count = int(np.count_nonzero(keep_mask))
    print("Watershedのマーカー数:", marker_count)
    print("Watershed後の領域数:", len(np.unique(labels)) - 1)

    return count, result_image