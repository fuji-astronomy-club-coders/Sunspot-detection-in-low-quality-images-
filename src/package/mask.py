
"""黒点候補を矩形領域によって除外するモジュール。"""

import cv2
import numpy as np


def selectMaskRegions(
    image: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """画像上で複数の矩形マスク領域を選択する。

    Enterで矩形を確定し、Escで選択を終了する。

    Args:
        image: 選択時に表示する画像。
    ---------
    Return
        選択した矩形領域のリスト。
        各要素は (x, y, width, height)。
    """
    if image.ndim != 2:
        raise ValueError("入力画像はグレースケール画像にしてください。")

    # 表示用に8bit画像へ変換
    display_image = np.empty(image.shape, dtype=np.uint8)

    cv2.normalize(
        image,
        display_image,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    regions: list[tuple[int, int, int, int]] = []
    window_name = "Select Mask Regions"

    try:
        while True:
            roi = cv2.selectROI(
                window_name,
                display_image,
                showCrosshair=True,
                fromCenter=False,
            )

            x, y, width, height = map(int, roi)

            # Escまたは空の選択で終了
            if width == 0 or height == 0:
                break

            regions.append((x, y, width, height))

            print(
                f"マスク領域 {len(regions)}: "
                f"x={x}, y={y}, width={width}, height={height}"
            )

    finally:
        cv2.destroyWindow(window_name)

    print(f"合計 {len(regions)} 個のマスク領域を選択しました。")

    return regions


def applyMask(
    labels: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    original_count: int,
) -> tuple[int, np.ndarray]:
    """矩形領域と重なる黒点候補を除外する。

    Args:
        labels: 黒点ごとに異なる整数ラベルを持つ画像。
        regions: (x, y, width, height) のリスト。
        original_count: マスク適用前の黒点候補数。
    --------
    Return
        count: マスク適用後の黒点候補数。
        masked_labels: マスク適用後のラベル画像。
    """
    if labels.ndim != 2:
        raise ValueError("ラベル画像は2次元配列にしてください。")

    masked_labels = labels.copy()
    height, width = labels.shape
    excluded_labels: set[int] = set()

    for x, y, roi_width, roi_height in regions:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + roi_width)
        y2 = min(height, y + roi_height)

        if x1 >= x2 or y1 >= y2:
            continue

        roi_labels = labels[y1:y2, x1:x2]
        overlapping_labels = np.unique(roi_labels)

        excluded_labels.update(
            int(label)
            for label in overlapping_labels
            if label != 0
        )

    if excluded_labels:
        masked_labels[np.isin(labels, list(excluded_labels))] = 0

    count = original_count - len(excluded_labels)

    print("除外した黒点数:", len(excluded_labels))
    print("マスク適用後の黒点数:", count)

    masked_image = np.where(
        masked_labels > 0,
        0,
        255,
    ).astype(np.uint8)

    return count, masked_image
