"""Count connected components in a thresholded image."""
import numpy as np
from scipy import ndimage


def count_connected_components(
    image: np.ndarray,
    threshold: float = 10000,
) -> tuple[int, np.ndarray]:
    """画像を二値化し、連結成分数とラベリング画像を返す。

    Args:
        image: 二値化前の2次元画像を表すNumPy配列。
        threshold: 二値化のしきい値。これ未満の画素を対象領域とする。
    Returns:
        count: 連結成分の個数
        binary_image: 二値化画像のNumpy配列。0か100かで二値化する。
    """
    # 二値化：背景を0、対象領域を100にする
    binary_image = (image < threshold).astype(np.uint8) * 100

    labeled_image, count = ndimage.label(binary_image)  # pyright: ignore[reportGeneralTypeIssues]

    return count, binary_image