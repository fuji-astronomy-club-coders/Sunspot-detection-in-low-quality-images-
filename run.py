"""設定ファイルに基づいて太陽画像処理を実行するエントリーポイント。"""

import argparse
import logging
import sys
from pathlib import Path

from src.main import main as run_processing
from src.package.batch_analysis import (
    BatchInputError,
    discover_batch_images,
    run_batch_analysis,
)
from src.package.comparison import (
    ComparisonInputError,
    prepare_comparison,
    run_comparison,
)
from src.package.config import PROJECT_ROOT, config
from src.package.qt_dialogs import select_directory, select_file
from src.package.run_logger import RunLogger

LOGGER = logging.getLogger("sunspot")


def main() -> int:
    """Run either the unchanged single-image workflow or comparison mode."""
    parser = argparse.ArgumentParser(description="太陽黒点検出")
    parser.add_argument(
        "--batch",
        nargs="?",
        const="",
        default=None,
        metavar="INPUT_FOLDER",
        help=(
            "フォルダー直下のTIFF/FITS画像をすべて一括解析する。"
            "省略時はフォルダー選択ダイアログを表示する。"
        ),
    )
    parser.add_argument(
        "--compare",
        nargs="?",
        const="",
        default=None,
        metavar="DATE_FOLDER",
        help=(
            "同じ日付の最大4観測所を比較する。日付フォルダーの親に sources.yaml が必要。"
            "リポジトリ外のフォルダーも指定可能。省略時はフォルダー選択ダイアログを表示する。"
        ),
    )
    arguments = parser.parse_args()

    if arguments.batch is not None and arguments.compare is not None:
        parser.error("--batch と --compare は同時に指定できません。")

    if arguments.batch is not None:
        input_folder = arguments.batch
        if not input_folder:
            default_folder = PROJECT_ROOT / "source_input"
            selected_folder = select_directory(
                title="一括解析するTIFF/FITS画像フォルダーを選択してください",
                directory=str(default_folder if default_folder.is_dir() else PROJECT_ROOT),
            )
            if selected_folder is None:
                print("一括解析フォルダーの選択がキャンセルされました。", file=sys.stderr)
                return 2
            input_folder = selected_folder

        selected_path = Path(input_folder).expanduser()
        if not selected_path.is_absolute():
            selected_path = PROJECT_ROOT / selected_path
        selected_path = selected_path.resolve()
        try:
            image_files = discover_batch_images(selected_path)
        except BatchInputError as error:
            print(f"一括解析入力エラー: {error}", file=sys.stderr)
            return 2

        run_logger = RunLogger(config)
        LOGGER.info("Selected batch input folder: %s", selected_path)
        try:
            summary = run_batch_analysis(image_files, config, run_logger)
        except BaseException as error:
            run_logger.fail(error)
            raise

        run_logger.finish(summary, status=summary["overall_status"])
        print(f"一括解析結果: {summary['overall_status']}")
        for row in summary["images"]:
            if row["status"] == "success":
                detail = (
                    f"検出={row['sunspot_count']}, "
                    f"除外={row['excluded_sunspot_count']}"
                )
            else:
                detail = row["error"]
            print(f"  - [{row['status']}] {row['file_name']}: {detail}")
        print(f"集計CSV: {summary['summary_csv']}")
        print(f"画像別出力: {summary['images_directory']}")
        return 1 if summary["overall_status"] != "success" else 0

    if arguments.compare is not None:
        date_folder = arguments.compare
        if not date_folder:
            source_input_folder = PROJECT_ROOT / "source_input"
            source_input_folder.mkdir(parents=True, exist_ok=True)
            selected_folder = select_directory(
                title="比較する日付フォルダーを選択してください（リポジトリ外も指定できます）",
                directory=str(source_input_folder),
            )
            if selected_folder is None:
                print("比較フォルダーの選択がキャンセルされました。", file=sys.stderr)
                return 2
            date_folder = selected_folder

        try:
            plan = prepare_comparison(date_folder)
        except ComparisonInputError as error:
            print(f"比較入力エラー: {error}", file=sys.stderr)
            return 2

        print(f"比較日付: {plan.comparison_date}")
        print("登録観測所:")
        for source in plan.sources:
            policy = source["orientation_policy"]
            if policy == "fixed":
                orientation_source = (
                    f"north_vector={source['north_vector']}, "
                    f"flip_x={source['flip_x']}"
                )
            else:
                orientation_source = "向きは日付別 comparison.yaml から取得"
            print(
                f"  - {source['source_id']} / {source['display_name']} "
                f"({policy}; {orientation_source})"
            )

        supplied_ids = {
            sample.source_id for sample in plan.samples if sample.source_id is not None
        }
        print("入力サンプル:")
        for sample in plan.samples:
            if sample.error is not None:
                print(
                    f"  - [skipped_invalid_input] {sample.source_id or '(unknown)'}: "
                    f"{sample.error}"
                )
                continue
            assert sample.source_id is not None
            assert sample.image_path is not None
            assert sample.pixel_scale is not None
            assert sample.orientation is not None
            source = next(
                item for item in plan.sources if item["source_id"] == sample.source_id
            )
            direction_source = (
                "sources.yaml"
                if source["orientation_policy"] == "fixed"
                else "comparison.yaml"
            )
            print(
                f"  - {sample.source_id}: {sample.image_path.name}; "
                f"{sample.pixel_scale:g} arcsec/pixel; "
                f"north_vector={sample.orientation.north_vector}; "
                f"north_angle_deg={sample.orientation.north_angle_deg:g}; "
                f"flip_x={sample.orientation.flip_x}; "
                f"方向の取得元={direction_source}"
            )
        for source in plan.sources:
            if source["source_id"] not in supplied_ids:
                print(
                    f"  - [not_provided] {source['source_id']} / "
                    f"{source['display_name']}"
                )

        if plan.mask_template_path is not None:
            print(f"共通マスクテンプレート: {plan.mask_template_path}")
        else:
            print("共通マスク: 最初に処理が成功した画像で選択します。")

        run_logger = RunLogger(config)
        LOGGER.info("Selected comparison date folder: %s", plan.date_folder)
        try:
            summary = run_comparison(plan, config, run_logger)
        except BaseException as error:
            run_logger.fail(error)
            raise

        run_logger.finish(
            {
                "comparison_status": summary["overall_status"],
                "successful_source_count": len(summary["successful_sources"]),
                "comparison_date": plan.comparison_date,
                "summary_path": str(run_logger.session_dir / "comparison_summary.json"),
            },
            status=summary["overall_status"],
        )

        print(f"比較結果: {summary['overall_status']}")
        for row in summary["samples"]:
            status = row["status"]
            label = row.get("source_id") or "(unknown)"
            detail = row.get("reason")
            if detail is None and "sunspot_count" in row:
                detail = f"検出数={row['sunspot_count']}"
            print(f"  - [{status}] {label}" + (f": {detail}" if detail else ""))
        print(f"比較サマリー: {run_logger.session_dir / 'comparison_summary.json'}")

        return 1 if summary["overall_status"] == "failed" else 0

    if config["Paths"]["input_image"] == "?Dialog?":
        selected_file = select_file(
            title="画像ファイルを選択してください",
            name_filter=(
                "Image files (*.tiff *.tif *.fits *.fit *.fts);;"
                "All files (*)"
            ),
        )
        if selected_file is None:
            raise ValueError("画像ファイルが選択されませんでした。")
        config["Paths"]["input_image"] = selected_file

    run_logger = RunLogger(config)
    try:
        count = run_processing(config=config, run_logger=run_logger)
    except BaseException as error:
        run_logger.fail(error)
        raise

    run_logger.finish({"sunspot_count": int(count)})
    return count


if __name__ == "__main__":
    exit_code = main()
    if "--compare" in sys.argv or "--batch" in sys.argv:
        raise SystemExit(exit_code)
