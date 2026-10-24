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

from src.package import black_top_hat, connection_count, gaussian_filter, sobel_filter, utils
from src.package import config as configMOD
from src.package import limb_darking_resolution as ldr


def main(config: dict) -> int:
    """Run the main process."""
    print("Starting main process...")
    print("\n\n===Config===")
    pprint(config)
    print("============\n\n")

    image_path = configMOD.resolve_path(config["Paths"]["input_image"])

    # 画像の明るさ（画素値）を取得し、変数image（numpy配列）に代入します。
    image = utils.load_image(image_path)    

    try:
        darkingConfig = config["LimbDarkingResolution"]
        darkingMethod = darkingConfig["method"]
        
        if darkingMethod == "polynomialFit":
            image = ldr.polynomialFit(
                image=image,
                **darkingConfig["polynomialFit"]
            )

        else:
            raise ValueError(f"Unknown limb darking resolution method: {darkingMethod}")
    except Exception as e:
        raise RuntimeError(f"[Fatal]: 周縁減光補正に失敗しました。:{e}") from e

    try:
        emphasisConfig = config["Emphasis"]
        emphasisMethod = emphasisConfig["method"]

        if emphasisMethod == "SobelFilter":
            image = sobel_filter.sobel_filter(
                image=image,
                return_derivs=False,
                **emphasisConfig["SobelFilter"]
            )
        elif emphasisMethod == "BlackTopHat":
            image = black_top_hat.black_top_hat(
                image=image,
                **emphasisConfig["BlackTopHat"]
            )
        elif emphasisMethod == "GaussianFilter":
            image = gaussian_filter.gaussian_high_pass_filter(
                image=image,
                **emphasisConfig["GaussianFilter"]
            )
        else:
            raise ValueError(f"Unknown emphasis method: {emphasisMethod}")
    except Exception as e:
        raise RuntimeError(f"[Fatal]: 強調処理に失敗しました。:{e}") from e

    try:
        extractionConfig = config["Extraction"]
        extractionMethod = extractionConfig["method"]

        if extractionMethod == "count_components":
            count, image = connection_count.count_connected_components(
                image,  # pyright: ignore[reportArgumentType]
                **extractionConfig["count_components"]
            )
        else:
            raise ValueError(f"Unknown extraction method: {extractionMethod}")
    except Exception as e:
        raise RuntimeError(f"[Fatal]: 連結成分の抽出に失敗しました。:{e}") from e
        
    print(bool(image))
    print("連結成分の個数：",count)

    return count

    """
    plt.imshow(image, origin="lower", cmap="gray")
    plt.colorbar(label="Pixel value")
    plt.show()
    """
