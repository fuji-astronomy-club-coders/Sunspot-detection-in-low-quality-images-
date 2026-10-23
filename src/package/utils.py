"""プロジェクト全体で使用する共通ユーティリティを提供するモジュール。

複数のモジュールから再利用される汎用的な処理を集約する。
特定の機能や実行モードに依存する処理は ``src/package/`` 配下に配置し、
本モジュールには共通性の高い処理のみを実装する。

アプリケーションのメイン処理は ``src/main.py`` に、
実行制御はプロジェクトルートの ``run.py`` に配置する。
"""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import tifffile
from astropy.io import fits


def load_image(image_path: str | Path) -> npt.NDArray[Any]:
    """TIFFまたはFITS画像を読み込み、画素値をNumPy配列で返す。

    Args:
        image_path: 読み込む画像ファイルのパス。

    Returns
    -------
        各画素の値を格納したNumPy配列。

    Raises
    ------
        ValueError: 対応していない画像形式の場合。
    """
    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        image = tifffile.imread(path)
    elif suffix in (".fits", ".fit", ".fts"):
        image = fits.getdata(path)
    else:
        raise ValueError(f"対応していない画像形式です: {path}")

    return np.asarray(image)

def closing_circle(image: np.ndarray, kernel: np.ndarray | int) -> np.ndarray:
    """
    楕円形またはカスタムカーネルを使用して、画像に対するモルフォロジー閉じ処理(Closing)を実行します。

    Parameters
    ----------
    image : np.ndarray
        処理対象の入力画像。
    kernel : np.ndarray または int
        楕円形構造要素のサイズ(整数の場合)、またはカスタム構造要素の行列(np.ndarrayの場合)。

    Returns
    -------
    np.ndarray
        モルフォロジー閉じ処理が適用された画像。

    Raises
    ------
    TypeError
        引数`kernel` が整数でも numpy 配列でもない場合。
    """
    if isinstance(kernel, int):
        adopt_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel, kernel)
        )
    elif isinstance(kernel, np.ndarray):
        adopt_kernel = kernel
    else:
        raise TypeError("Invalid kernel type. Expected int or np.ndarray.")

    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, adopt_kernel)

    return closed