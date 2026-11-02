"""周縁減光を補正するプログラム"""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy import dtype, floating, ndarray

# 先輩が作った関数をインポート
from src.module.MIN2ver2 import MIN2_ignore_sunspots as min2

version = "0.0.1"
def polynomialFit(image:np.ndarray, alpha: float = 0.85, brightness_boost: float = 1.1) -> np.ndarray:
    """太陽像の半径方向の輝度プロファイルを用いて周縁減光を補正する。

    Parameters
    ----------
    image : numpy.ndarray
        2次元の入力太陽画像。
    alpha : float, optional
        補正前画像と平坦化画像を混合する割合。
    brightness_boost : float, optional
        補正後画像に適用する明るさ倍率。

    Returns
    -------
    numpy.ndarray
        周縁減光を補正した uint16 画像。

    Raises
    ------
    ValueError
        太陽円の推定に失敗した場合。
    """
    # 画像の読み込み

    h, w = image.shape
    img_float: ndarray[Any, dtype[floating]] = image.astype(np.float32)

    # 1. 先輩の関数（MIN2_ignore_sunspots）を使って正確な中心と半径を取得する
    try:
        (cx, cy), max_radius = min2(img_float)
    except Exception as e:
        raise ValueError(f"関数の実行に失敗しました: {e}") from e


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
    corrected = img_float * np.float32(1.0 - alpha) + flattened * np.float32(alpha)
    corrected = corrected * brightness_boost

    # 太陽の外側（黒背景）は元の状態を維持
    corrected[~mask] = img_float[~mask]
    corrected = np.clip(corrected, 0, 2**16 - 1).astype(np.uint16)

    return corrected


if __name__ == "__main__":

    input_dir = '.'
    # 「img」から始まるtiffファイルだけを自動でまとめて選ぶ
    image_paths = list(Path(input_dir).glob('img*.tiff'))
    
    if not image_paths:
        print("対象の画像ファイルが見つかりませんでした")
    else:
        # 保存用のフォルダを自動で作る
        output_dir = Path('corrected_images')
        output_dir.mkdir(exist_ok=True)

        print(f"合計 {len(image_paths)} 枚の画像を処理します...")

        eroCount = 0  # エラー画像のカウントを初期化
        # 1枚ずつ順番に一括処理！
        for path in image_paths:
            filename = Path(path).name
            output_path = output_dir / f'after_{filename}'
            
            print(f"処理中: {filename} ...")
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"画像が見つかりません: {path}")
                eroCount+=1
                continue

            corrected_image = polynomialFit(image=img, alpha=0.85, brightness_boost=1.1)
            if not cv2.imwrite(str(output_path), corrected_image):
                print(f"画像の保存に失敗しました: {output_path}")
                eroCount+=1
                continue
            
        print(f"{len(image_paths) - eroCount}/{len(image_paths)}件 の画像の自動処理と保存が成功しました！")