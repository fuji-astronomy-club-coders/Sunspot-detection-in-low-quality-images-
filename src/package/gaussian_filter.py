"""ガウシアン背景差し引きを行うモジュール。"""
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

version = 1.00

def gaussian_high_pass_filter(image:np.ndarray,sigmaSize:int) -> np.ndarray:
    """ガウシアン平滑化した背景を入力画像から差し引く。

    Parameters
    ----------
    image : numpy.ndarray
        背景成分を差し引く入力画像。
    sigmaSize : int
        ガウシアンフィルタのシグマサイズ

    Returns
    -------
    numpy.ndarray[float32]
        float32 に変換した入力画像から、`sigmaSize`で平滑化した背景を差し引いた配列。
    """
    # 元画像を浮動小数点型に変換
    image_float = image.astype(np.float32)

    # 背景画像を作成
    background = gaussian_filter(image_float, sigma=sigmaSize)

    # 背景を差し引く
    result = image_float - background

    return result

if __name__ == "__main__":
    # --------------------------------------------------
    # 設定
    # --------------------------------------------------
    SIGMA = 20 #シグマの大きさ
    INPUT_PATH = Path(r"C:\Users\2025005585\Desktop\python\raw_data\2025-08-30-0437_9-CapObj - コピー\2025-08-30-0437_9-CapObj - コピー") # 入力画像のパス
    IMAGE_EXT = ".tiff" # 対象にする画像の拡張子
    OUTPUT_PATH = Path(r"C:\Users\2025005585\Desktop\python\Gaussian_high_pass_filter") # 出力画像の保存先
    OUTPUT_NAME = "GF_" + str(SIGMA) + "sigma_'25-8-30_1fr_sample2"


    # フォルダ内の画像をファイル名順に取得
    image_paths = sorted(INPUT_PATH.glob(f"*{IMAGE_EXT}"))
    
    if not image_paths:
        raise FileNotFoundError(
            f"画像が見つかりませんでした: {INPUT_PATH}"
        )
    
    # 最初の1枚を読み込む
    input_path = image_paths[0]
    image = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"画像を読み込めませんでした: {input_path}"
        )
        print(f"処理対象の画像: {input_path.name}")
    
    result = gaussian_high_pass_filter(image,sigmaSize=SIGMA)

    # 表示用に0～255へ正規化
    result_min = float(np.min(result))
    result_max = float(np.max(result))

    if result_max > result_min:
        result_display = (
            (result - result_min)
            / (result_max - result_min)
            * 255
        ).astype(np.uint8)
    else:
        result_display = np.zeros_like(result, dtype=np.uint8)

    # 保存先フォルダを作成
    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # 正規化した背景差し引き画像を保存
    output_path = OUTPUT_PATH / f"{OUTPUT_NAME}.png"
    success = cv2.imwrite(str(output_path), result_display)

    if not success:
        raise OSError(
            f"画像を保存できませんでした: {output_path}"
        )

    print(f"処理が完了しました: {output_path}")