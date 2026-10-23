import numpy as np
from scipy import ndimage

def count_connected_components(
    binary_image: np.ndarray,
) -> tuple[int, np.ndarray]:
    """二値画像の連結成分数とラベリング画像を取得する。

    Args:
        binary_image: 二値化された2次元NumPy配列。
            背景を0、対象領域を0以外の値で表す。

    Returns:
        連結成分数とラベリング画像のタプル。
        背景のラベルは0、各連結成分には1から始まる
        異なる整数のラベルが割り当てられる。
    """
    labeled_image, count = ndimage.label(binary_image)

    return count, labeled_image