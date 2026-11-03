# CSV 出力ガイド

このドキュメントでは、太陽黒点検出処理が出力する CSV の項目と読み方を説明します。

## 1. 出力場所

CSV は実行ごとに作成されるセッションフォルダー直下へ保存されます。

```text
<Paths.output_folder>/
└─ session_YYYYMMDD_HHMMSS_xxxxxx/
   ├─ sunspot_statistics.csv
   └─ sunspot_exclusion_list.csv
```

- `sunspot_statistics.csv`: minArea とマスクの適用後に残った黒点
- `sunspot_exclusion_list.csv`: 通常mask、Rmask、minArea によって除外された黒点

どちらのファイルも UTF-8 BOM 付き（`utf-8-sig`）、カンマ区切りです。Excel などの表計算ソフトで開く場合は、文字コードを UTF-8 として読み込んでください。検出対象がない場合も、ヘッダー行のみの CSV が出力されます。

## 2. 座標系と基本的な用語

すべての座標は、**画像の左上を原点とする0始まりのピクセル座標**です。

- X: 右方向に増加
- Y: 下方向に増加
- 座標 `(0, 0)`: 画像の左上の画素

**重心（centroid）** は、黒点を構成する画素の X 座標・Y 座標それぞれの平均値です。小数点以下6桁に丸めて出力します。

**外接矩形（bbox）** は、黒点を構成するすべての画素を囲む最小の矩形です。`bbox_x_px` と `bbox_y_px` は矩形の左上座標です。幅・高さには両端の画素を含めるため、次の式で求めます。

```text
bbox_width_px  = 最大X座標 - 最小X座標 + 1
bbox_height_px = 最大Y座標 - 最小Y座標 + 1
```

**画素面積（area_px）** は、黒点を構成する画素の実数です。外接矩形の面積（幅 × 高さ）とは異なります。外接矩形の中に黒点以外の画素が含まれる場合、通常は `area_px` の方が小さくなります。

## 3. `sunspot_statistics.csv` — 残った黒点

マスク処理によって除外されず、最終結果に残った黒点を1行につき1件記録します。

| 列名               | 内容                                  |
| ---------------- | ----------------------------------- |
| `sunspot_id`     | CSV内での通し番号。1から始まります。                |
| `label`          | 抽出処理が黒点候補に付与したラベル番号。元のラベル画像における値です。 |
| `centroid_x_px`  | 黒点の重心のX座標（ピクセル）。                    |
| `centroid_y_px`  | 黒点の重心のY座標（ピクセル）。                    |
| `bbox_x_px`      | 外接矩形の左上X座標（ピクセル）。                   |
| `bbox_y_px`      | 外接矩形の左上Y座標（ピクセル）。                   |
| `bbox_width_px`  | 外接矩形の幅（ピクセル）。                       |
| `bbox_height_px` | 外接矩形の高さ（ピクセル）。                      |
| `area_px`        | 黒点を構成する画素数。                         |

### 出力例

以下は列の意味を示すための架空の値です。

```csv
sunspot_id,label,centroid_x_px,centroid_y_px,bbox_x_px,bbox_y_px,bbox_width_px,bbox_height_px,area_px
1,3,245.625,180.125,230,165,32,31,512
2,7,510.2,302.8,501,294,19,18,176
```

この例の1行目は、ラベル `3` の黒点について、重心が `(245.625, 180.125)`、外接矩形が左上 `(230, 165)`・幅32・高さ31、黒点を構成する画素が512個であることを表します。

## 4. `sunspot_exclusion_list.csv` — 除外された黒点

マスク領域との重なり、または `minArea` フィルターによって除外された黒点を1行につき1件記録します。黒点の形状・位置に関する列は、残った黒点の CSV と同じ意味です。`minArea` で除外された候補もフィルター前のラベル画像から統計値を取得するため、マスク除外と同じ CSV で確認できます。

| 列名                         | 内容                                                 |
| -------------------------- | -------------------------------------------------- |
| `sunspot_id`               | 除外一覧 CSV 内での通し番号。1から始まります。残った黒点の CSV の番号とは対応しません。  |
| `label`                    | 抽出処理が黒点候補に付与したラベル番号。残った黒点の CSV と照合する場合は、この値を使用します。 |
| `centroid_x_px`            | 除外された黒点の重心のX座標（ピクセル）。                              |
| `centroid_y_px`            | 除外された黒点の重心のY座標（ピクセル）。                              |
| `bbox_x_px`                | 外接矩形の左上X座標（ピクセル）。                                  |
| `bbox_y_px`                | 外接矩形の左上Y座標（ピクセル）。                                  |
| `bbox_width_px`            | 外接矩形の幅（ピクセル）。                                      |
| `bbox_height_px`           | 外接矩形の高さ（ピクセル）。                                     |
| `area_px`                  | 黒点を構成する画素数。                                        |
| `excluded_by_factor`       | 除外要因。通常maskは `mask`、半径マスクは `Rmask`、面積フィルターは `minArea` です。複数要因は `;` 区切りです。 |
| `exclusion_stage`           | 除外段階。`mask`、`connected_components`、`watershed` のいずれかです。 |
| `min_area_threshold_px`     | minArea の最小面積（画素数）。mask による除外では空欄です。                 |
| `excluded_by_mask_ids`     | 除外に関係したマスク番号。複数ある場合はセミコロン（`;`）区切りです。               |
| `excluded_by_mask_regions` | 除外に関係した矩形マスク領域の詳細を格納した JSON 配列です。 |
| `radial_distance_px` | Rmask における太陽中心から黒点重心までの距離（ピクセル）です。 |
| `radial_limit_px` | Rmask の除外境界までの距離（ピクセル）です。 |
| `radial_ratio` | 太陽半径に対する Rmask 境界の比率です。 |
| `radial_center_x_px / radial_center_y_px` | MIN2 が求めた太陽中心の座標です。Rmask 以外では空欄です。 |

### マスク番号と領域情報

マスク番号は、マスク選択画面で選択した順番に対応する **1始まりの番号**です。`excluded_by_mask_ids` は一覧で該当マスクをすばやく確認するための列です。

`excluded_by_mask_regions` は、マスクごとの位置と大きさを JSON 配列で記録します。各要素のキーは次のとおりです。

| JSON キー   | 内容                 |
| --------- | ------------------ |
| `mask_id` | マスク番号（1始まり）。       |
| `x`       | マスク矩形の左上X座標（ピクセル）。 |
| `y`       | マスク矩形の左上Y座標（ピクセル）。 |
| `width`   | マスク矩形の幅（ピクセル）。     |
| `height`  | マスク矩形の高さ（ピクセル）。    |

1つの黒点が複数のマスク領域と重なっている場合、**該当するすべてのマスク番号と領域情報**を記録します。たとえば、マスク1とマスク3に重なった黒点では、次のようになります。

```text
excluded_by_mask_ids:
1;3

excluded_by_mask_regions:
[{"mask_id":1,"x":100,"y":80,"width":120,"height":90},{"mask_id":3,"x":180,"y":120,"width":100,"height":75}]
```

上記の座標・大きさは説明用の架空の値です。JSON 配列の値も画像左上原点・0始まりのピクセル座標です。

### Rmask の距離情報

Rmask は MIN2 が推定した太陽中心・半径を基準に、中心からの距離が `solar_radius × radial_ratio` 以上となる黒点を除外します。`radial_distance_px` は候補重心の距離、`radial_limit_px` は除外境界です。Rmask と矩形maskの両方に該当した黒点は、`excluded_by_factor` に `mask;Rmask` のように両方の要因を記録します。

### 除外要因の見方

- `excluded_by_factor` に `mask` が含まれる場合: 選択した矩形マスクによる除外です。マスク番号と範囲は `excluded_by_mask_ids` / `excluded_by_mask_regions` で確認できます。
- `excluded_by_factor` に `Rmask` が含まれる場合: MIN2 の太陽中心からの距離が半径比の境界以上となったため除外されています。距離と境界は `radial_distance_px` / `radial_limit_px` で確認できます。
- `excluded_by_factor=minArea`: `area_px` が `min_area_threshold_px` 未満だったため除外されています。
- `exclusion_stage=connected_components`: 連結成分抽出直後の面積判定で除外されています。
- `exclusion_stage=watershed`: Watershed による分割後の面積判定で除外されています。

`label` は処理段階ごとに付け直されるため、異なる `exclusion_stage` の行同士をラベル番号だけで照合しないでください。

## 5. CSV を読むときのポイント

1. **最終的に残った黒点の個数を確認する**  
   `sunspot_statistics.csv` のデータ行数が、CSVに出力された残存黒点数です。ヘッダー行は数えません。

2. **除外された黒点と理由を確認する**  
   `sunspot_exclusion_list.csv` の `excluded_by_factor` を確認します。`mask` の場合は `excluded_by_mask_ids` と `excluded_by_mask_regions`、`Rmask` の場合は `radial_distance_px` と `radial_limit_px`、`minArea` の場合は `area_px` と `min_area_threshold_px` を比較してください。

3. **黒点の位置と大きさを画像上で確認する**  
   `centroid_x_px`・`centroid_y_px` が重心位置、`bbox_x_px`・`bbox_y_px`・`bbox_width_px`・`bbox_height_px` が外接矩形です。座標は0始まりなので、画像ビューアーによって表示座標の数え方が異なる場合は注意してください。

4. **通し番号とラベル番号を混同しない**  
   `sunspot_id` は各 CSV の中だけで使う行番号です。2つの CSV を照合するときは `label` を使ってください。`label` は同じ実行セッション内での識別用であり、別の実行結果との間で同じ黒点を示す保証はありません。

5. **外接矩形の面積と黒点の画素面積を区別する**  
   `bbox_width_px × bbox_height_px` は黒点を囲む矩形の面積です。`area_px` は黒点そのものを構成する画素数です。

## 6. 除外一覧の JSON をプログラムで読む例

Python では `csv.DictReader` と `json.loads` を使って、マスク領域の詳細を取り出せます。

```python
import csv
import json
from pathlib import Path

csv_path = Path("sunspot_exclusion_list.csv")

with csv_path.open(encoding="utf-8-sig", newline="") as file:
    for row in csv.DictReader(file):
        mask_ids = [
            int(value)
            for value in row["excluded_by_mask_ids"].split(";")
            if value
        ]
        mask_regions = json.loads(row["excluded_by_mask_regions"])

        print(
            "label:", row["label"],
            "mask IDs:", mask_ids,
            "mask regions:", mask_regions,
        )
```

`excluded_by_mask_ids` が空の場合は区切り文字による分割結果も空になります。`excluded_by_mask_regions` は JSON 配列として読み込めます。
