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

import logging
from pprint import pformat
from typing import Any

import numpy as np
from numpy import dtype, ndarray
from numpy._typing._shape import _AnyShape

from src.package import (
    black_top_hat,
    connection_count,
    gaussian_filter,
    mask,
    sobel_filter,
    utils,
    watershed,
)
from src.package import config as configMOD
from src.package import limb_darking_resolution as ldr
from src.package.run_logger import RunLogger
from src.package.qt_preview import show_final_preview

LOGGER = logging.getLogger("sunspot.main")


def _format_analysis_parameters(config: dict[str, Any]) -> str:
    """選択された処理方式で実際に使用する解析パラメータを整形する。

    Parameters
    ----------
    config : dict[str, Any]
        アプリケーション設定。

    Returns
    -------
    str
        処理段階ごとに空行で区切った解析パラメータ。
    """
    sections: list[str] = []

    limb_config = config["LimbDarkingResolution"]
    limb_method = limb_config["chosen_method"]
    sections.append(
        "\n".join(
            [
                "[周縁減光補正]",
                f"method: {limb_method}",
                pformat(limb_config.get(limb_method, {}), sort_dicts=False),
            ]
        )
    )

    emphasis_config = config["Emphasis"]
    emphasis_method = emphasis_config["chosen_method"]
    sections.append(
        "\n".join(
            [
                "[強調処理]",
                f"method: {emphasis_method}",
                pformat(emphasis_config.get(emphasis_method, {}), sort_dicts=False),
            ]
        )
    )

    extraction_config = config["Extraction"]
    extraction_method = extraction_config["chosen_method"]
    extraction_lines = [
        "[黒点候補抽出]",
        f"method: {extraction_method}",
        "countComponents:",
        pformat(extraction_config.get("countComponents", {}), sort_dicts=False),
    ]
    if extraction_method == "watershed":
        extraction_lines.extend(
            [
                "watershed:",
                pformat(extraction_config.get("watershed", {}), sort_dicts=False),
            ]
        )
    sections.append("\n".join(extraction_lines))

    return "\n\n".join(sections)


def main(config: dict, run_logger: RunLogger | None = None) -> int:
    """設定に従って太陽画像の補正、強調、黒点候補抽出を実行する。

    Parameters
    ----------
    config : dict
        入力画像、出力先、各画像処理モジュールの設定。

    Returns
    -------
    int
        マスク適用後に残った黒点候補数。
    """
    LOGGER.info("Starting main process...")
    # 実際に使用する解析パラメータを処理段階ごとに空行で区切って表示する。
    LOGGER.info(
        "=== Analysis Parameters ===\n%s\n===========================",
        _format_analysis_parameters(config),
    )

    image_path = configMOD.resolve_path(config["Paths"]["input_image"])
    output_root = (
        run_logger.session_dir
        if run_logger is not None
        else configMOD.resolve_path(config["Paths"]["output_folder"])
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # 画像の明るさ（画素値）を取得し、変数image（numpy配列）に代入します。
    LOGGER.info("Input image: %s", image_path)
    image = utils.loadImage(image_path)
    LOGGER.info("Loaded image: shape=%s, dtype=%s", image.shape, image.dtype)
    try:
        image_metadata = utils.collect_image_metadata(image_path, image)
        if run_logger is not None:
            run_logger.record_event("input_image_metadata", image_metadata)
        else:
            LOGGER.debug("Input image metadata: %s", image_metadata)
    except Exception as error:
        # Metadata collection is supplementary and must not prevent image processing.
        LOGGER.warning(
            "入力画像のメタデータを取得できませんでした。画像処理は継続します: %s",
            error,
        )
        LOGGER.debug("画像メタデータ取得時の詳細:", exc_info=True)

    moduleVersions = {}
    raw_image = image
    min_area_exclusions: list[tuple[np.ndarray, list[int], str, int]] = []

    try:
        darkingConfig = config["LimbDarkingResolution"]
        darkingMethod = darkingConfig["chosen_method"]
        LOGGER.info("Limb darkening method: %s", darkingMethod)

        svFolder = output_root

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
    limb_corrected_image = image

    try:
        emphasisConfig = config["Emphasis"]
        emphasisMethod = emphasisConfig["chosen_method"]
        LOGGER.info("Emphasis method: %s", emphasisMethod)

        svFolder = output_root

        if emphasisMethod == "SobelFilter":
            moduleVersions["Emphasis"] = sobel_filter.version

            imgName = f"SobelFilter_kernel{emphasisConfig['SobelFilter']['kernelSize']}.png"

            image = sobel_filter.sobel_filter(  # pyright: ignore[reportAssignmentType]
                image=image,
                return_derivs=False,
                **emphasisConfig["SobelFilter"]
            )
        elif emphasisMethod == "BlackTopHat":

            imgName = f"BlackTopHat_kernel{emphasisConfig['BlackTopHat']['kernelSize']}.png"

            moduleVersions["Emphasis"] = black_top_hat.version
            image = black_top_hat.black_top_hat(
                image=image,
                **emphasisConfig["BlackTopHat"]
            )
        elif emphasisMethod == "GaussianFilter":

            imgName = f"GaussianFilter_sigma{emphasisConfig['GaussianFilter']['sigmaSize']}.png"

            moduleVersions["Emphasis"] = gaussian_filter.version
            image = gaussian_filter.gaussian_high_pass_filter(
                image=image, **emphasisConfig["GaussianFilter"]
            )
        else:
            raise ValueError(f"Unknown emphasis method: {emphasisMethod}")
        
    except Exception as e:
        raise RuntimeError(f"[Fatal]: 強調処理に失敗しました。:{e}") from e
    
    utils.svImage_withInfo(image,{"config": config, "moduleVersions": moduleVersions},svFolder / imgName, )

    # マスク適用前の強調画像も従来どおり保存し、後からマスク適用版と区別できるようにする。
    emphasis_img_name = imgName
    emphasized_image = image

    try:
        extractionConfig = config["Extraction"]
        extractionMethod = extractionConfig["chosen_method"]
        LOGGER.info("Extraction method: %s", extractionMethod)

        svFolder = output_root

        if extractionMethod == "countComponents":
            imgName = f"countComponents_c{extractionConfig['countComponents']['c']}_threshold{extractionConfig['countComponents']['space_threshold']}_minArea{extractionConfig['countComponents']['min_area']}.png"

            count, image, binary_image, connected_component_details = (
                connection_count.countConnectedComponents(
                    image,  # pyright: ignore[reportArgumentType]
                    **extractionConfig["countComponents"],
                    return_details=True,
                )
            )
            if connected_component_details["excluded_labels"]:
                min_area_exclusions.append(
                    (
                        connected_component_details["unfiltered_labels"],
                        connected_component_details["excluded_labels"],
                        "connected_components",
                        connected_component_details["min_area"],
                    )
                )
        elif extractionMethod == "watershed":
            imgName = f"watershed_c{extractionConfig['countComponents']['c']}_threshold{extractionConfig['countComponents']['space_threshold']}_minArea{extractionConfig['countComponents']['min_area']}_minDistance{extractionConfig['watershed']['min_distance']}.png"

            count, image, binary_image, connected_component_details = (
                connection_count.countConnectedComponents(
                    image,  # pyright: ignore[reportArgumentType]
                    **extractionConfig["countComponents"],
                    return_details=True,
                )
            )
            if connected_component_details["excluded_labels"]:
                min_area_exclusions.append(
                    (
                        connected_component_details["unfiltered_labels"],
                        connected_component_details["excluded_labels"],
                        "connected_components",
                        connected_component_details["min_area"],
                    )
                )

            count, image, binary_image, watershed_details = watershed.watershedCount(
                image,
                extractionConfig["countComponents"]["min_area"],
                extractionConfig["watershed"]["min_distance"],
                return_details=True,
            )
            if watershed_details["excluded_labels"]:
                min_area_exclusions.append(
                    (
                        watershed_details["unfiltered_labels"],
                        watershed_details["excluded_labels"],
                        "watershed",
                        watershed_details["min_area"],
                    )
                )
        else:
            raise ValueError(f"Unknown extraction method: {extractionMethod}")

        regions = mask.selectMaskRegions(
            binary_image, config.get("MaskDialog", {}).get("display")
        )

        # 選択した除外領域を0で埋め、強調画像のマスク適用版も保存する。
        # 0は Black Top-Hat / Sobel の応答を抑制し、Gaussian 高域通過画像でも
        # 中立値として扱える。元の強調画像は変更せず、別ファイルに出力する。
        masked_emphasized_image = emphasized_image.copy()
        mask_output_regions = []
        image_height, image_width = masked_emphasized_image.shape[:2]
        for x, y, width, height in regions:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(image_width, int(x + width))
            y2 = min(image_height, int(y + height))
            if x1 >= x2 or y1 >= y2:
                continue
            masked_emphasized_image[y1:y2, x1:x2] = 0
            mask_output_regions.append(
                {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )

        masked_emphasis_path = svFolder / (
            emphasis_img_name.removesuffix(".png") + "_masked.png"
        )
        utils.svImage_withInfo(
            masked_emphasized_image,
            {
                "config": config,
                "moduleVersions": moduleVersions,
                "mask": {
                    "applied": True,
                    "fill_value": 0,
                    "coordinate_origin": "top-left",
                    "regions": mask_output_regions,
                },
            },
            masked_emphasis_path,
        )
        LOGGER.info("マスク適用済み強調画像を保存しました: %s", masked_emphasis_path)
        if run_logger is not None:
            run_logger.record_event(
                "mask_regions_selected",
                {
                    "image_width": int(binary_image.shape[1]),
                    "image_height": int(binary_image.shape[0]),
                    "selected_region_count": len(regions),
                    "regions": [
                        {
                            "index": index,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                        }
                        for index, (x, y, width, height) in enumerate(regions, start=1)
                    ],
                },
            )

        sunspot_labels = binary_image.copy()
        count, image, mask_details = mask.applyMask(
            binary_image,  # pyright: ignore[reportArgumentType]
            regions,
            count,
            return_details=True,
        )
        if run_logger is not None:
            run_logger.record_event("mask_applied", mask_details)

        statistics_path = output_root / "sunspot_statistics.csv"
        exported_count = utils.save_sunspot_statistics_csv(
            sunspot_labels,
            mask_details["excluded_candidate_labels"],
            statistics_path,
        )
        LOGGER.info(
            "Saved sunspot statistics: %s (%d spots)",
            statistics_path,
            exported_count,
        )
        exclusion_list_path = output_root / "sunspot_exclusion_list.csv"
        excluded_count = utils.save_sunspot_exclusion_csv(
            sunspot_labels,
            mask_details["regions"],
            exclusion_list_path,
            min_area_exclusions=min_area_exclusions,
        )
        LOGGER.info(
            "Saved sunspot exclusion list: %s (%d excluded spots)",
            exclusion_list_path,
            excluded_count,
        )
        if run_logger is not None:
            run_logger.record_event(
                "sunspot_statistics_exported",
                {
                    "csv_path": str(statistics_path),
                    "spot_count": exported_count,
                    "exclusion_list_csv_path": str(exclusion_list_path),
                    "excluded_spot_count": excluded_count,
                    "coordinate_origin": "top-left",
                    "coordinate_indexing": "zero-based pixels",
                    "excluded_by_mask": [
                        {
                            "label": int(label),
                            "mask_ids": [
                                int(region["index"])
                                for region in mask_details["regions"]
                                if int(label) in region["overlapping_candidate_labels"]
                            ],
                        }
                        for label in mask_details["excluded_candidate_labels"]
                    ],
                    "excluded_by_factor": {
                        "mask": len(mask_details["excluded_candidate_labels"]),
                        "minArea": sum(
                            len(excluded_labels)
                            for _labels, excluded_labels, _stage, _threshold
                            in min_area_exclusions
                        ),
                    },
                },
            )
    except Exception as e:
        raise RuntimeError(f"[Fatal]: 連結成分の抽出に失敗しました。:{e}") from e
    
    utils.svImage_withInfo(image,{"result": {"count": count}, "config": config, "moduleVersions": moduleVersions},svFolder / imgName, )

    binary_image = image

    # 検証用画像を出力
    utils.output_image(
        raw_image,
        limb_corrected_image,
        emphasized_image,
        binary_image,
        config,
        image_path,
        count,
        output_dir=output_root,
    )
    # 対話的な最終確認は設定で有効化した場合のみ PyQt ダイアログで表示する。
    if config.get("Display", {}).get("show_final_image", False):
        invalid_sunspot_mask = np.isin(
            sunspot_labels,
            mask_details["excluded_candidate_labels"],
        )
        show_final_preview(
            image,
            regions,
            config.get("Display", {}),
            invalid_sunspot_mask,
        )
    

    LOGGER.info("Processing completed: %d sunspots detected", count)
    return count
