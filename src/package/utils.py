"""プロジェクト全体で使用する共通ユーティリティを提供するモジュール。

複数のモジュールから再利用される汎用的な処理を集約する。
特定の機能や実行モードに依存する処理は ``src/package/`` 配下に配置し、
本モジュールには共通性の高い処理のみを実装する。

アプリケーションのメイン処理は ``src/main.py`` に、
実行制御はプロジェクトルートの ``run.py`` に配置する。
"""
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import tifffile
from astropy.io import fits
from PIL import Image as PlImage
from PIL import PngImagePlugin


def loadImage(image_path: str | Path) -> npt.NDArray[Any]:
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
    if not path.exists():
        raise ValueError(f"存在しないパスです: {path}")

    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        image = tifffile.imread(path)
    elif suffix in (".fits", ".fit", ".fts"):
        image = fits.getdata(path)
    else:
        raise ValueError(f"対応していない画像形式です: {suffix}")
    if image is None:
        raise ValueError(f"failed to load the image: {image_path}")
    return np.asarray(image)

def svImage_withInfo(image: np.ndarray, info: dict | str, path: str | Path) -> None:
    """画像をメタデータ(PngInfo)とともに保存します。

    Parameters
    ----------
    image : np.ndarray
        保存する画像データ(np.ndarray)。
    info : dict | str
        画像に付与するメタデータやパラメータ。
        辞書型の場合は各キー・バリューがPNGのテキストチャンクとして保存されます。
        文字列の場合は {"info": 文字列} として保存されます。
    path : str | Path
        保存先のファイルパス。
    
    Returns
    -------
    None
    """
    file_path = Path(path)
    # 保存先の親ディレクトリを作成
    file_path.parent.mkdir(parents=True, exist_ok=True)

    im = PlImage.fromarray(image)

    # メタデータ（PngInfo）を作成
    metadata = PngImagePlugin.PngInfo()
    valid_info = info if isinstance(info, dict) else {"info": info}

    for key, value in valid_info.items():
        # 辞書型の場合はJSON文字列に変換、それ以外は文字列に変換
        param_str = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
        
        # キーと値のペアでパラメータを追加
        metadata.add_text(key, param_str)

    try:
        # パラメータを含めて保存
        im.save(file_path, pnginfo=metadata)
        print(f"画像を保存しました: {file_path}")
    except Exception as e:
        print(f"画像の保存に失敗しました: {file_path} (エラー: {e})")


def closingCircle(image: np.ndarray, kernel: np.ndarray | int) -> np.ndarray:
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