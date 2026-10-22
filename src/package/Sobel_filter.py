"""16 bit画像にSobelフィルターを適用し、結果を画像として保存する。"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

version = 1.00

# --------------------------------------------------
# 設定
# --------------------------------------------------

INPUT_DIR = Path(r"C:\Users\2025005585\Desktop\python\raw_data\2025-08-30-0437_9-CapObj - コピー\2025-08-30-0437_9-CapObj - コピー")
IMAGE_EXT = ".tiff" # 対象にする画像の拡張子
OUTPUT_DIR = Path(r"C:\Users\2025005585\Desktop\python\Sobel_filter")
KSIZE = 3
OUTPUT_NAME_GX = "Sobel_gx_'25-8-30_1f_sample2.png"
OUTPUT_NAME_GY = "Sobel_gy_'25-8-30_1f_sample2.png"
OUTPUT_NAME_G = "Sobel_magnitude_'25-8-30_1f_sample2.png"


# --------------------------------------------------
# 表示用画像への変換
# --------------------------------------------------

def normalize_signed_image(image: np.ndarray) -> np.ndarray:
    """符号付き微分画像を表示用の 8 bit 画像に正規化する。

    Parameters
    ----------
    image : numpy.ndarray[np.float32]
        正負の値を含む微分画像。sobel_filter 関数の出力 gx または gy が該当する。

    Returns
    -------
    numpy.ndarray
        8bit範囲に変換した画像。入力がすべて 0 の場合は 128 の画像。
    """
    max_abs = float(np.max(np.abs(image)))

    if max_abs == 0:
        return np.full(image.shape, 128, dtype=np.uint8)

    normalized = image / max_abs * 127 + 128
    return np.clip(normalized, 0, 255).astype(np.uint8)


def normalize_magnitude_image(image: np.ndarray) -> np.ndarray:
    """勾配の大きさを表示用の 8 bit 画像に min-max正規化する。

    Parameters
    ----------
    image : numpy.ndarray
        勾配の大きさを表す非負の配列。sobel_filter 関数の出力 g が該当する。

    Returns
    -------
    numpy.ndarray
        8bit範囲に min-max正規化をした 画像。最大値が 0 の場合は黒画像。
    """
    max_value = float(np.max(image))

    if max_value == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = image / max_value * 255
    return np.clip(normalized, 0, 255).astype(np.uint8)


# --------------------------------------------------
# メイン処理
# --------------------------------------------------

def sobel_filter(image:np.ndarray,return_derivs:bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画像の水平方向・垂直方向の Sobel 微分と勾配の大きさを計算する。

    Parameters
    ----------
    image : numpy.ndarray
        Sobel 演算を適用する入力画像。
    return_derivs : bool, optional
        `defaut=False` Trueなら`gx`と`gy`も返します。

    Returns
    -------
    g : numpy.ndarray[np.float32]
            ``gx`` と ``gy`` から計算した勾配の大きさ。向きを持たないので、絶対値のユークリッド距離になる。
    gx : numpy.ndarray[np.float32]
        水平方向の微分画像
    gy : numpy.ndarray[np.float32]
        垂直方向の微分画像
    """
    # 微分を計算
    image_float = image.astype(np.float32)

    gx = cv2.Sobel(
        image_float, cv2.CV_32F,
        dx=1, dy=0, ksize=KSIZE,
    )

    gy = cv2.Sobel(
        image_float, cv2.CV_32F,
        dx=0, dy=1, ksize=KSIZE,
    )

    # 勾配の大きさ
    g = cv2.magnitude(gx, gy)
    
    if return_derivs:
        return g, gx, gy # ! 順番を変えました10/22 !
    else:
        return g

if __name__ == "__main__":
        # フォルダ内のTIFF画像を取得
    image_paths = sorted(INPUT_DIR.glob(f"*{IMAGE_EXT}"))

    if not image_paths:
        raise FileNotFoundError(
            f"画像が見つかりませんでした: {INPUT_DIR}"
        )

    # 最初の1枚を読み込む
    input_path = image_paths[0]
    print(f"読み込み対象: {input_path}")
    print(f"ファイルが存在するか: {input_path.is_file()}")
    print(f"ファイルサイズ: {input_path.stat().st_size if input_path.is_file() else 'ファイルなし'}")
    #image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    try:
        with Image.open(input_path) as pil_image:
            print(f"画像形式: {pil_image.format}")
            print(f"画像モード: {pil_image.mode}")
            print(f"画像サイズ: {pil_image.size}")

            image = np.array(pil_image)

        print(f"読み込み成功: dtype={image.dtype}, shape={image.shape}")

    except Exception as e:
        print(f"Pillowでも読み込めませんでした: {e}")
        raise


    if image is None:
        raise FileNotFoundError(
            f"画像を読み込めませんでした: {input_path}"
        )

    # 16 bitグレースケール画像か確認
    if image.ndim != 2 or image.dtype != np.uint16:
        raise ValueError(
            f"16 bitグレースケール画像ではありません: "
            f"shape={image.shape}, dtype={image.dtype}"
        )

    # 出力フォルダを作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gx, gy, g = sobel_filter(image)

    # 表示用の8 bit画像に変換
    gx_display = normalize_signed_image(gx)
    gy_display = normalize_signed_image(gy)
    g_display = normalize_magnitude_image(g)

    # 画像を保存
    print("出力フォルダ:", OUTPUT_DIR.resolve())
    print("出力ファイル:", (OUTPUT_DIR / OUTPUT_NAME_GX).resolve())
    print("出力ファイル名:", OUTPUT_NAME_GX)
    cv2.imwrite(str(OUTPUT_DIR / OUTPUT_NAME_GX), gx_display)
    cv2.imwrite(str(OUTPUT_DIR / OUTPUT_NAME_GY), gy_display)
    cv2.imwrite(str(OUTPUT_DIR / OUTPUT_NAME_G), g_display)

    print(f"処理が完了しました。出力先: {OUTPUT_DIR}")
