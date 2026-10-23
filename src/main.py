"""アプリケーションのメイン処理を定義するモジュール。

アプリケーション全体の処理を統括し、各機能モジュールを連携させる。
具体的な機能の実装は ``src/package/`` 配下のモジュールに分離し、
本モジュールではメイン処理の構成と実行順序を管理する。

実行入口はプロジェクトルートの ``run.py`` とし、
実行設定や実行モード、実行コンテキストに応じた制御は ``run.py`` に委ねる。

主な役割:
    - アプリケーションのメイン処理の定義
    - 各機能モジュールの呼び出しと連携
    - アプリケーション全体の処理フローの管理
    - ``run.py`` から受け取った設定に基づく処理の実行
"""

from pprint import pprint

import matplotlib.pyplot as plt

from src.package.config import resolve_path
from src.package.utils import load_image

from src.package.connection_count import count_connected_components


def main(config: dict) -> None:
    """Run the main process."""
    print("Hello World")
    print("===Config===")
    pprint(config)

    # 読み込む画像のパスを指定してください
    image_path = resolve_path(config["paths"]["input_image"])

    # 画像の明るさ（画素値）を取得し、変数image（numpy配列）に代入します。
    image = load_image(image_path)

    count, image = count_connected_components(image)

    # ここから先のコードは正しく画像が読み込まれているかの確認用です。必要に応じてコメントアウトしてください。
    print(image)

    plt.imshow(image, origin="lower", cmap="gray")
    plt.colorbar(label="Pixel value")
    plt.show()
