"""周縁減光を補正するプログラム"""
import cv2
import numpy as np

# 1. 処理したい画像のファイル名を設定
input_image_path = 'img00000001.tiff'   # 元の太陽画像のファイル名
output_image_path = 'after-Limb-darkening.tiff' # 補正後の保存先ファイル名

# 補正の強さ（0.7〜1.0。1.0で完全にフラット）
alpha = 1.0
# 全体の明るさ調整（1.0でそのまま）
brightness_boost = 100.0

# 画像の読み込み（グレースケール）
img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("画像が見つかりませんでした。ファイル名やパスを確認してください。")
    exit()

h, w = img.shape
img_float = img.astype(np.float32)

# 2. 【最小二乗法】を使って太陽の正確な中心と半径を求める
_, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    # 輪郭の座標を抽出
    pts = largest_contour.reshape(-1, 2).astype(np.float32)
    X = pts[:, 0]
    Y = pts[:, 1]
    
    # 最小二乗法で円のパラメータを計算 (x^2 + y^2 + Ax + By + C = 0)
    A_mat = np.column_stack([X, Y, np.ones_like(X)])
    B_vec = -(X**2 + Y**2)
    params, _, _, _ = np.linalg.lstsq(A_mat, B_vec, rcond=None)
    A, B, C = params
    
    # 係数から中心と半径を算出
    cx = -A / 2
    cy = -B / 2
    max_radius = int(np.sqrt(cx**2 + cy**2 - C))
    cx, cy = int(cx), int(cy)
    print(f"最小二乗法で求めた中心: ({cx}, {cy}), 半径: {max_radius}")
else:
    # 万が一輪郭が見つからない場合の保険
    cx, cy = w // 2, h // 2
    max_radius = int(min(cx, cy, w - cx, h - cy))
    print("輪郭が見つからなかったため、画像の真ん中を基準にします。")
# 3. 極座標変換（円を四角い展開図に変換）
polar_img = cv2.warpPolar(
    img_float, 
    (360, max_radius), 
    (cx, cy), 
    max_radius, 
    cv2.WARP_POLAR_LINEAR
)

# 4. 横方向（角度方向）の平均をとって「端に向かってどれくらい暗くなっているか」を調べる
profile = np.mean(polar_img, axis=1, keepdims=True)
profile = np.clip(profile, 1, 255) # 0割（ゼロ除算）を防ぐ安全策

# 5. 周縁減光のぶんだけ割り算して、全体を均等な明るさ（フラット）にする
mean_brightness = np.mean(profile)
corrected_polar = (polar_img / profile) * mean_brightness
corrected_polar = np.clip(corrected_polar, 0, 255).astype(np.float32)

# 6. 四角い展開図を、もう一度元の円形の太陽に戻す
corrected_img = cv2.warpPolar(
    corrected_polar, 
    (w, h), 
    (cx, cy), 
    max_radius, 
    cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
)

# 7. 画像として保存
cv2.imwrite(output_image_path, corrected_img.astype(np.uint8))
print(f"補正が完了しました！保存された画像: {output_image_path}")