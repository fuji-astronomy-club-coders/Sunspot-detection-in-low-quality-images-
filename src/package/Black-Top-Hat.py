"""太陽画像にBlack Top-Hat変換を適用し、結果を画像ファイルに保存する。"""
from pathlib import Path

import cv2
import numpy as np

version = 1.00

# =========================
# 設定
# =========================
KERNEL_SIZE = 15 # 構造要素（カーネル）の大きさ
INPUT_PATH = Path(r"C:\Users\2025005585\Desktop\python\raw_data\2025-08-30-0437_9-CapObj - コピー\2025-08-30-0437_9-CapObj - コピー") # 入力画像のパス
IMAGE_EXT = ".tiff" # 対象にする画像の拡張子
OUTPUT_PATH = Path(r"C:\Users\2025005585\Desktop\python\Black Top-Hat") # 出力画像の保存先
OUTPUT_NAME = "BTH_" + str(KERNEL_SIZE) + "karnel_'25-8-30_1fr_sample2"
CLOSING_NAME = "BTH_" + str(KERNEL_SIZE) + "karnel_'25-8-30_1fr_closing_sample2"


# =========================
# Black Top-Hat変換
# =========================

def black_top_hat(image:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy配列を受け取り、クロージング画像とBlack Top-Hat画像を返す。"""
    # カーネルを作成(カーネルサイズ四方に納まる楕円形)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    # クロージング処理
    closing = cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel,
        #anchor=(7, 8),
    )

    # Black Top-Hat変換
    result = cv2.morphologyEx(
        image,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    return closing, result

if __name__ == "__main__":
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

    closing, result = black_top_hat(image)

        # 保存先フォルダを作成
    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # クロージング画像を保存
    closing_path = OUTPUT_PATH / f"{CLOSING_NAME}_closing.png"
    success = cv2.imwrite(str(closing_path), closing)

    if not success:
        raise OSError(
            f"画像を保存できませんでした: {closing_path}"
        )

    print(f"処理が完了しました: {closing_path}")

    # Black Top-Hat変換画像を保存
    output_path = OUTPUT_PATH / f"{OUTPUT_NAME}.png"
    success_top_hat = cv2.imwrite(str(output_path), result)

    if not success_top_hat:
        raise OSError(
            f"画像を保存できませんでした: {output_path}"
        )

    print(f"処理が完了しました: {output_path}")
