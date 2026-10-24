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

from pathlib import Path
from pprint import pprint
from typing import Any

from numpy import dtype, ndarray
from numpy._typing._shape import _AnyShape

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
    image = utils.loadImage(image_path)    

    moduleVersions = {}

    try:
        darkingConfig = config["LimbDarkingResolution"]
        darkingMethod = darkingConfig["method"]

        svFolder = Path(config["path"]["outputfolder"]) / config["path"]["folder_names"]["LimbDarkingResolution"]

        if darkingMethod == "polynomialFit":
            moduleVersions["LimbDarkingResolution"] = ldr.version
            image: ndarray[_AnyShape, dtype[Any]] = ldr.polynomialFit(
                image=image,
                **darkingConfig["polynomialFit"]
            )
            
            imgName = f"polynomialFit_alpha{darkingConfig["polynomialFit"]["alpha"]}_boost{darkingConfig["polynomialFit"]["brightness_boost"]}.png"

        # elif darkingMethod == "~~~~":
        # ...
        
        else:
            raise ValueError(f"Unknown limb darking resolution method: {darkingMethod}")
        

    except Exception as e:
        raise RuntimeError(f"[Fatal]: 周縁減光補正に失敗しました。:{e}") from e

    utils.svImage_withInfo(image,{"config": config, "moduleVersions": moduleVersions},svFolder / imgName, )

    try:
        emphasisConfig = config["Emphasis"]
        emphasisMethod = emphasisConfig["method"]

        svFolder = Path(config["path"]["outputfolder"]) / config["path"]["folder_names"]["Emphasis"]

        if emphasisMethod == "SobelFilter":
            moduleVersions["Emphasis"] = sobel_filter.version

            imgName = f"SobelFilter_{emphasisConfig['SobelFilter']['ksize']}.png"

            image = sobel_filter.sobel_filter(  # pyright: ignore[reportAssignmentType]
                image=image,
                return_derivs=False,
                **emphasisConfig["SobelFilter"]
            )
        elif emphasisMethod == "BlackTopHat":

            imgName = f"BlackTopHat_{emphasisConfig['BlackTopHat']['kernel_size']}.png"

            moduleVersions["Emphasis"] = black_top_hat.version
            image = black_top_hat.black_top_hat(
                image=image,
                **emphasisConfig["BlackTopHat"]
            )
        elif emphasisMethod == "GaussianFilter":

            imgName = f"GaussianFilter_{emphasisConfig['GaussianFilter']['sigma']}.png"

            moduleVersions["Emphasis"] = gaussian_filter.version
            image = gaussian_filter.gaussian_high_pass_filter(
                image=image,
                **emphasisConfig["GaussianFilter"]
            )
        else:
            raise ValueError(f"Unknown emphasis method: {emphasisMethod}")
        
    except Exception as e:
        raise RuntimeError(f"[Fatal]: 強調処理に失敗しました。:{e}") from e
    
    utils.svImage_withInfo(image,{"config": config, "moduleVersions": moduleVersions},svFolder / imgName, )

    try:
        extractionConfig = config["Extraction"]
        extractionMethod = extractionConfig["method"]

        svFolder = config["path"]["outputfolder"] / config["path"]["folder_names"]["Extraction"]

        if extractionMethod == "countComponents":

            imgName = f"countComponents_{extractionConfig['countComponents']['connectivity']}.png"

            count, image = connection_count.countConnectedComponents(
                image,  # pyright: ignore[reportArgumentType]
                **extractionConfig["countComponents"]
            )
        else:
            raise ValueError(f"Unknown extraction method: {extractionMethod}")
    except Exception as e:
        raise RuntimeError(f"[Fatal]: 連結成分の抽出に失敗しました。:{e}") from e
    
    utils.svImage_withInfo(image,{"result": {"count": count}, "config": config, "moduleVersions": moduleVersions},svFolder / imgName, )

    print(bool(image))
    print("連結成分の個数：",count)

    return count

    """
    plt.imshow(image, origin="lower", cmap="gray")
    plt.colorbar(label="Pixel value")
    plt.show()
    """
