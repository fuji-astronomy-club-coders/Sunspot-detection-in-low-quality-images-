"""周縁減光を補正するプログラム"""
import glob
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy import dtype, floating, ndarray

# 先輩が作った関数をインポート
from MIN2ver2 import MIN2_ignore_sunspots


def polynomialFit(input_image_path, output_image_path, alpha=0.85, brightness_boost=1.1):
    """周縁減光を距離マップと先輩の関数で補正する関数"""
    # 画像の読み込み
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"画像が見つかりません: {input_image_path}")
        return

    h, w = img.shape
    img_float: ndarray[Any, dtype[floating]] = img.astype(np.float32)

    # 1. 先輩の関数（MIN2_ignore_sunspots）を使って正確な中心と半径を取得する
    try:
        cx, cy, max_radius = MIN2_ignore_sunspots(img)  # pyright: ignore[reportAssignmentType]
        cx, cy, max_radius = int(cx), int(cy), int(max_radius)  # pyright: ignore[reportArgumentType]
    except Exception as e:
        print(f"関数の実行に失敗しました: {e}")
        cx, cy = w // 2, h // 2
        max_radius = int(min(cx, cy, w - cx, h - cy))

    # 2. 画像全体について「中心からの距離（マップ）」を計算する
    Y_indices, X_indices = np.indices((h, w), dtype=np.float32)
    distance_map = np.sqrt((X_indices - cx)**2 + (Y_indices - cy)**2)

    # 3. 距離ごとの平均輝度（減光プロファイル）を計算する
    mask = distance_map <= max_radius
    max_r_int = int(max_radius)

    profile = np.zeros(max_r_int + 1, dtype=np.float32)
    for r in range(max_r_int + 1):
        ring_mask = (distance_map >= r - 0.5) & (distance_map < r + 0.5) & mask
        if np.any(ring_mask):
            profile[r] = np.mean(img_float[ring_mask])
        else:
            profile[r] = profile[r-1] if r > 0 else 1.0

    # 0割を防ぐ処理 ＆ 滑らかにする
    profile = np.clip(profile, 1, 255)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (15, 1), 0).flatten()

    # 4. 背景モデルを復元してフラット化する
    distance_clipped = np.clip(distance_map, 0, max_r_int).astype(int)
    background_model = profile[distance_clipped]

    mean_brightness = np.mean(profile[profile > 0])
    flattened = (img_float / background_model) * mean_brightness

    # 5. 補正の強さ（alpha）と明るさ調整を反映
    corrected = img_float * (1.0 - alpha) + flattened * alpha
    corrected = corrected * brightness_boost

    # 太陽の外側（黒背景）は元の状態を維持
    corrected[~mask] = img_float[~mask]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # 6. 保存
    cv2.imwrite(output_image_path, corrected)
    print(f"補正完了: {output_image_path}")


if __name__ == "__main__":
    # 「img」から始まるtiffファイルだけを自動でまとめて選ぶ
    image_paths = glob.glob('img*.tiff')
    
    if not image_paths:
        print("対象の画像ファイルが見つかりませんでした")
    else:
        # 保存用のフォルダを自動で作る
        output_dir = 'corrected_images'
        os.makedirs(output_dir, exist_ok=True)

        print(f"合計 {len(image_paths)} 枚の画像を処理します...")

        # 1枚ずつ順番に一括処理！
        for path in image_paths:
            filename = Path(path).name
            output_path = os.path.join(output_dir, f'after_{filename}')
            
            print(f"処理中: {filename} ...")
            polynomialFit(input_image_path=path, output_image_path=output_path)
            
        print("すべての画像の自動処理が完了しました！")