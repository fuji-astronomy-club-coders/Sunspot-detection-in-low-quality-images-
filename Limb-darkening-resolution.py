"""周縁減光を補正するプログラム"""
import cv2
import numpy as np

# 1. ファイルの設定
input_image_path = 'img00000001.tiff'
output_image_path = 'after-Limb-darkening.tiff'

# 補正の強さ（0.7〜1.0。1.0で完全にフラット）
alpha = 0.85
# 全体の明るさ調整（1.0でそのまま）
brightness_boost = 1.1

# 画像の読み込み
img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("画像が見つかりません。")
    exit()

h, w = img.shape
img_float = img.astype(np.float32)

# 2. 最小二乗法で正確な中心と半径を求める
_, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    pts = largest_contour.reshape(-1, 2).astype(np.float32)
    X, Y = pts[:, 0], pts[:, 1]
    
    A_mat = np.column_stack([X, Y, np.ones_like(X)])
    B_vec = -(X**2 + Y**2)
    params, _, _, _ = np.linalg.lstsq(A_mat, B_vec, rcond=None)
    A, B, C = params
    
    cx = -A / 2
    cy = -B / 2
    max_radius = int(np.sqrt(cx**2 + cy**2 - C))
    cx, cy = int(cx), int(cy)
else:
    cx, cy = w // 2, h // 2
    max_radius = int(min(cx, cy, w - cx, h - cy))

# 3. 画像全体について「中心からの距離（マップ）」を計算する
Y_indices, X_indices = np.indices((h, w), dtype=np.float32)
distance_map = np.sqrt((X_indices - cx)**2 + (Y_indices - cy)**2)

# 4. 距離ごとの平均輝度（減光プロファイル）を効率よく計算する
# 太陽の半径の内側だけで平均をとる
mask = distance_map <= max_radius
max_r_int = int(max_radius)

# 距離ごとの明るさを入れる配列を作る
profile = np.zeros(max_r_int + 1, dtype=np.float32)
for r in range(max_r_int + 1):
    # 半径 r のまわりの画素を抽出して平均をとる（少し幅を持たせる）
    ring_mask = (distance_map >= r - 0.5) & (distance_map < r + 0.5) & mask
    if np.any(ring_mask):
        profile[r] = np.mean(img_float[ring_mask])
    else:
        profile[r] = profile[r-1] if r > 0 else 1.0

# 0割を防ぐ処理 ＆ 滑らかにする
profile = np.clip(profile, 1, 255)
# 輪郭付近のノイズを防ぐため軽く平滑化
profile = cv2.GaussianBlur(profile.reshape(-1, 1), (15, 1), 0).flatten()

# 5. 各画素の距離に対応する「理想の明るさ（背景モデル）」を2D画像として復元する
# 距離マップの値をインデックスとして使って、背景画像を作る
distance_clipped = np.clip(distance_map, 0, max_r_int).astype(int)
background_model = profile[distance_clipped]

# 全体の平均輝度を基準にして割り算する
mean_brightness = np.mean(profile[profile > 0])
flattened = (img_float / background_model) * mean_brightness

# 太陽の外側（黒い背景部分）は元画像の黒を維持する処理
# 6. 補正の強さ（alpha）と明るさブーストを反映
corrected = img_float * (1.0 - alpha) + flattened * alpha
corrected = corrected * brightness_boost

# 太陽の外側は黒に戻す、かつ0〜255に収める
corrected[~mask] = img_float[~mask]
corrected = np.clip(corrected, 0, 255).astype(np.uint8)

# 7. 保存
cv2.imwrite(output_image_path, corrected)
print(f"補正が完了しました！保存ファイル: {output_image_path}")