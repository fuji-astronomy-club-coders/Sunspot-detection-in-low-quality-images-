"""プロジェクト全体で使用する共通ユーティリティを提供するモジュール。

複数のモジュールから再利用される汎用的な処理を集約する。
特定の機能や実行モードに依存する処理は ``src/package/`` 配下に配置し、
本モジュールには共通性の高い処理のみを実装する。

例:
    - 共通処理の提供
    - データ形式の変換や検証などの汎用処理
    - 文字列・パス・日時などの共通操作
    - 複数のモジュールで重複する処理の集約

アプリケーションのメイン処理は ``src/main.py`` に、
実行制御はプロジェクトルートの ``run.py`` に配置する。
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tifffile
from astropy.io import fits


def load_image(image_path: str | Path) -> npt.NDArray[Any]:
    """TIFFまたはFITS画像を読み込み、画素値をNumPy配列で返す。

    Args:
        image_path: 読み込む画像ファイルのパス。

    Returns:
        各画素の値を格納したNumPy配列。

    Raises:
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
