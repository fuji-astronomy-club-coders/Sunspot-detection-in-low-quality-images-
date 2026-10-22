"""プロジェクト全体で使用する共通ユーティリティを提供するモジュール。

複数のモジュールから再利用される汎用的な処理を集約する。
特定の機能や実行モードに依存する処理は ``src/package/`` 配下に配置し、
本モジュールには共通性の高い処理のみを実装する。

アプリケーションのメイン処理は ``src/main.py`` に、
実行制御はプロジェクトルートの ``run.py`` に配置する。
"""
import cv2
import numpy as np


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