"""プロジェクト全体で使用する共通ユーティリティを提供するモジュール。

複数のモジュールから再利用される汎用的な処理を集約する。
特定の機能や実行モードに依存する処理は ``src/package/`` 配下に配置し、
本モジュールには共通性の高い処理のみを実装する。

アプリケーションのメイン処理は ``src/main.py`` に、
実行制御はプロジェクトルートの ``run.py`` に配置する。
"""

from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tifffile
from astropy.io import fits
from matplotlib import font_manager


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
        adopt_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    elif isinstance(kernel, np.ndarray):
        adopt_kernel = kernel
    else:
        raise TypeError("Invalid kernel type. Expected int or np.ndarray.")

    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, adopt_kernel)

    return closed


available_fonts = {font.name for font in font_manager.fontManager.ttflist}


def save_processing_summary(
    images: list[tuple[str, np.ndarray]],
    config: dict,
    image_path: Path,
    count: int,
    output_path: Path,
) -> None:
    """output_imageから呼び出される、画像のレイアウトを決める関数"""
    # 日本語フォントを追加
    for font_name in ("Yu Gothic", "Meiryo", "MS Gothic"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.8, 2.2, 2.2],
        hspace=0.35,
        wspace=0.15,
    )

    # 上部：基本情報
    ax_info = fig.add_subplot(grid[0, :])
    ax_info.axis("off")
    ax_info.text(
        0.01,
        0.95,
        "Solar Image Processing Summary",
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    ax_info.text(
        0.01,
        0.62,
        f"Input file: {image_path.name}\nGenerated: {generated_at}\nConnected components: {count}",
        fontsize=11,
        va="top",
    )
    ax_info.text(
        0.48,
        0.62,
        "設定値:\n" + pformat(config, sort_dicts=False),
        fontsize=8,
        va="top",
    )

    # 下部：4段階の画像
    for i, (title, image) in enumerate(images):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        # 二値画像は0と255をそのまま白黒表示する
        if i == 3:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image, cmap="gray")

    fig.subplots_adjust(top=0.96, bottom=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def output_image(
    raw_image: np.ndarray,
    limb_corrected_image: np.ndarray,
    emphasized_image: np.ndarray,
    binary_image: np.ndarray,
    config: dict[str, Any],
    image_path: Path | str,
    count: int,
) -> None:
    """処理段階ごとの画像と実行情報を1枚の画像に保存する。"""
    images = [
        ("1. Raw Image", raw_image),
        ("2. Limb Darkening Corrected", limb_corrected_image),
        ("3. Enhanced Image", emphasized_image),
        ("4. Binary Image", binary_image),
    ]

    output_path = Path("output") / f"{Path(image_path).stem}_summary.png"

    save_processing_summary(
        images=images,
        config=config,
        image_path=Path(image_path),
        count=count,
        output_path=output_path,
    )

    print(f"処理結果レポートを保存しました: {output_path}")
