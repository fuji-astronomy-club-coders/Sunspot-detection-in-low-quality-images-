"""設定ファイルに基づいて太陽画像処理を実行するエントリーポイント。"""

import argparse
import sys

from src.main import main as run_processing
from src.package.comparison import (
    ComparisonInputError,
    prepare_comparison,
    run_comparison,
)
from src.package.config import PROJECT_ROOT, config
from src.package.qt_dialogs import select_directory, select_file
from src.package.run_logger import RunLogger


def main() -> int:
    """Run either the unchanged single-image workflow or comparison mode."""
    parser = argparse.ArgumentParser(description="太陽黒点検出")
    parser.add_argument(
        "--compare",
        nargs="?",
        const="",
        default=None,
        metavar="DATE_FOLDER",
        help="同じ日付の最大4観測所を比較する。省略時はフォルダー選択ダイアログを表示する。",
    )
    arguments = parser.parse_args()

    if arguments.compare is not None:
        date_folder = arguments.compare
        if not date_folder:
            selected_folder = select_directory(
                title="比較する日付フォルダーを選択してください",
                directory=str(PROJECT_ROOT / "source_input"),
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

        run_logger = RunLogger(config)
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
            }
        )
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
    if "--compare" in sys.argv:
        raise SystemExit(exit_code)
