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
from pathlib import Path
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
from src.package.solar_coordinates import SolarTransform
from src.package.version_info import get_application_version
from src.package.qt_preview import (
    build_invalid_sunspot_mask,
    show_final_preview,
)

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

    mask_config = config.get("Mask", {})
    sections.append(
        "\n".join(
            [
                "[マスク除外]",
                f"boundary_policy: {mask_config.get('boundary_policy', 'overlap')}",
                pformat(mask_config, sort_dicts=False),
            ]
        )
    )

    return "\n\n".join(sections)


def process_image(
    config: dict,
    run_logger: RunLogger | None = None,
    *,
    output_dir: str | Path | None = None,
    solar_geometry_override: tuple[tuple[float, float], float] | None = None,
    mask_polygons: list[np.ndarray] | None = None,
    mask_preview_images: list[tuple[str, np.ndarray, SolarTransform]] | None = None,
    mask_editor_transform: SolarTransform | None = None,
    export_comparison_stages: bool = False,
) -> int:
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
        Path(output_dir)
        if output_dir is not None
        else (
            run_logger.session_dir
            if run_logger is not None
            else configMOD.resolve_path(config["Paths"]["output_folder"])
        )
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

    moduleVersions: dict[str, Any] = {"Application": get_application_version()}
    raw_image = image
    solar_geometry = solar_geometry_override
    min_area_exclusions: list[tuple[np.ndarray, list[int], str, int]] = []

    try:
        darkingConfig = config["LimbDarkingResolution"]
        darkingMethod = darkingConfig["chosen_method"]
        LOGGER.info("Limb darkening method: %s", darkingMethod)

        svFolder = output_root

        if darkingMethod == "polynomialFit":
            moduleVersions["LimbDarkingResolution"] = ldr.version
            if solar_geometry is None:
                solar_geometry = ldr.find_solar_geometry(image)
            image: ndarray[_AnyShape, dtype[Any]] = ldr.polynomialFit(
                image=image,
                solar_geometry=solar_geometry,
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
            moduleVersions["Extraction"] = {"countComponents": connection_count.version}
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
            moduleVersions["Extraction"] = {
                "countComponents": connection_count.version,
                "watershed": watershed.version,
            }
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

        mask_settings = config.get("Mask", {})
        selection_method = str(mask_settings.get("selection_method", "rectangle"))
        if selection_method not in ("rectangle", "connected_component"):
            raise ValueError(
                "Mask.selection_method は rectangle または connected_component を指定してください。"
            )

        regions: list[tuple[int, int, int, int]] = []
        component_selection: dict[str, Any] | None = None
        selected_component_labels: list[int] = []
        selected_component_polygons: list[np.ndarray] = []
        if mask_polygons is None:
            if selection_method == "connected_component":
                component_selection = mask.selectMaskComponents(
                    limb_corrected_image,
                    binary_image,
                    config.get("MaskDialog", {}).get("display"),
                    preview_images=mask_preview_images,
                    editor_transform=mask_editor_transform,
                )
                selected_component_labels = component_selection["selected_labels"]
                selected_component_polygons = [
                    np.asarray(polygon, dtype=np.float64)
                    for polygon in component_selection["polygons"]
                ]
            else:
                regions = mask.selectMaskRegions(
                    limb_corrected_image,
                    config.get("MaskDialog", {}).get("display"),
                    preview_images=mask_preview_images,
                    editor_transform=mask_editor_transform,
                )

        # 選択した除外領域を0で埋め、強調画像のマスク適用版も保存する。
        # 0は Black Top-Hat / Sobel の応答を抑制し、Gaussian 高域通過画像でも
        # 中立値として扱える。元の強調画像は変更せず、別ファイルに出力する。
        masked_emphasized_image = emphasized_image.copy()
        radial_settings = mask_settings.get("Rmask", {})
        radial_mask_output = None
        if radial_settings.get("enabled", False):
            if solar_geometry is None:
                raise ValueError("Rmask を使用するには MIN2 の太陽中心・半径が必要です。")
            (center_x, center_y), solar_radius = solar_geometry
            radius_ratio = float(radial_settings.get("radius_ratio", 0.85))
            if not np.isfinite(radius_ratio) or radius_ratio <= 0.0:
                raise ValueError("Mask.Rmask.radius_ratio は正の有限値にしてください。")
            radial_limit = solar_radius * radius_ratio
            if not np.isfinite(radial_limit):
                raise ValueError("Mask.Rmask の除外境界半径が有限値の範囲を超えています。")
            image_y, image_x = np.indices(masked_emphasized_image.shape[:2], dtype=np.float32)
            radial_area = np.hypot(image_x - center_x, image_y - center_y) >= radial_limit
            masked_emphasized_image[radial_area] = 0
            radial_mask_output = {
                "enabled": True,
                "center_x_px": float(center_x),
                "center_y_px": float(center_y),
                "solar_radius_px": float(solar_radius),
                "radius_ratio": radius_ratio,
                "threshold_radius_px": float(radial_limit),
            }
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
        if selected_component_labels:
            selected_pixels = np.isin(binary_image, selected_component_labels)
            masked_emphasized_image[selected_pixels] = 0
        if mask_polygons is not None:
            polygon_coverage = mask.polygon_coverage_mask(
                masked_emphasized_image.shape[:2], mask_polygons
            )
            masked_emphasized_image[polygon_coverage] = 0

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
                    "components": (
                        component_selection["components"]
                        if component_selection is not None
                        else []
                    ),
                    "polygons": (
                        [polygon.tolist() for polygon in mask_polygons]
                        if mask_polygons is not None
                        else [polygon.tolist() for polygon in selected_component_polygons]
                    ),
                    "radial_mask": radial_mask_output,
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
                    "selection_method": selection_method,
                    "selected_region_count": (
                        len(regions)
                        if component_selection is None
                        else len(selected_component_labels)
                    ),
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
                    "components": (
                        component_selection["components"]
                        if component_selection is not None
                        else []
                    ),
                    "selected_component_labels": selected_component_labels,
                    "polygons": (
                        [polygon.tolist() for polygon in mask_polygons]
                        if mask_polygons is not None
                        else (
                            [polygon.tolist() for polygon in selected_component_polygons]
                            if component_selection is not None
                            else None
                        )
                    ),
                },
            )

        sunspot_labels = binary_image.copy()
        boundary_policy = mask_settings.get("boundary_policy", "overlap")
        if mask_polygons is not None:
            _, _, mask_details = mask.applyPolygonMask(
                sunspot_labels,
                mask_polygons,
                count,
                return_details=True,
                boundary_policy=boundary_policy,
            )
        elif component_selection is not None:
            _, _, mask_details = mask.applyComponentMask(
                sunspot_labels,
                selected_component_labels,
                count,
                return_details=True,
            )
        else:
            _, _, mask_details = mask.applyMask(
                sunspot_labels,
                regions,
                count,
                return_details=True,
                boundary_policy=boundary_policy,
            )
            mask_details["selection_method"] = selection_method
        if mask_polygons is not None:
            for region in mask_details["regions"]:
                region["selection_method"] = selection_method
        normal_mask_exclusions = list(mask_details["excluded_candidate_labels"])

        radial_details: dict[str, Any] = {
            "enabled": False,
            "excluded_candidate_count": 0,
            "excluded_candidate_labels": [],
            "excluded_candidates": [],
        }
        radial_exclusions: list[int] = []
        if radial_settings.get("enabled", False):
            if solar_geometry is None:
                raise ValueError("Rmask を使用するには MIN2 の太陽中心・半径が必要です。")
            _, _, radial_details = mask.applyRadialMask(
                sunspot_labels,
                count,
                solar_geometry[0],
                solar_geometry[1],
                float(radial_settings.get("radius_ratio", 0.85)),
                boundary_policy=boundary_policy,
                return_details=True,
            )
            radial_exclusions = list(radial_details["excluded_candidate_labels"])

        combined_exclusions = sorted(
            set(normal_mask_exclusions) | set(radial_exclusions)
        )
        masked_labels = sunspot_labels.copy()
        if combined_exclusions:
            masked_labels[np.isin(masked_labels, combined_exclusions)] = 0
        count = max(0, int(count) - len(combined_exclusions))
        image = np.where(masked_labels > 0, 0, 255).astype(np.uint8)

        mask_details.update(
            {
                "boundary_policy": mask_details.get("boundary_policy", boundary_policy),
                "selection_method": selection_method,
                "excluded_by_factor": {
                    "mask": normal_mask_exclusions,
                    "Rmask": radial_exclusions,
                },
                "radial_mask": radial_details,
                "excluded_candidate_labels": combined_exclusions,
                "excluded_candidate_count": len(combined_exclusions),
                "remaining_candidate_count": count,
            }
        )
        if run_logger is not None:
            run_logger.record_event("module_versions", moduleVersions)
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
            radial_mask_details=radial_details,
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
                                if int(label) in region["excluded_candidate_labels"]
                            ],
                        }
                        for label in normal_mask_exclusions
                    ],
                    "excluded_by_rmask": radial_details.get("excluded_candidates", []),
                    "excluded_by_factor": {
                        "mask": len(normal_mask_exclusions),
                        "Rmask": len(radial_exclusions),
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

    if export_comparison_stages:
        stage_directory = output_root / ".comparison_stages"
        stage_directory.mkdir(parents=True, exist_ok=True)
        stage_images = {
            "raw": raw_image,
            "limb_corrected": limb_corrected_image,
            "emphasized": emphasized_image,
            "binary": binary_image,
        }
        for stage_name, stage_image in stage_images.items():
            np.save(
                stage_directory / f"{stage_name}.npy",
                np.asarray(stage_image),
                allow_pickle=False,
            )
        LOGGER.info("比較サマリー用の解析段階画像を保存しました: %s", stage_directory)

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
        invalid_sunspot_mask = build_invalid_sunspot_mask(
            sunspot_labels,
            normal_mask_exclusions,
            radial_exclusions,
        )
        show_final_preview(
            limb_corrected_image,
            regions,
            config.get("Display", {}),
            invalid_sunspot_mask,
            solar_geometry=solar_geometry,
            rmask_radius_ratio=float(
                config.get("Mask", {}).get("Rmask", {}).get("radius_ratio", 0.85)
            ),
            mask_polygons=(
                mask_polygons
                if mask_polygons is not None
                else selected_component_polygons
            ),
        )
    

    LOGGER.info("Processing completed: %d sunspots detected", count)
    return count



def main(config: dict, run_logger: RunLogger | None = None) -> int:
    """Run the standard single-image workflow without comparison-specific options."""
    return process_image(config=config, run_logger=run_logger)
