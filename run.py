"""設定ファイルに基づいて太陽画像処理を実行するエントリーポイント。"""

from src.main import main as run_processing
from src.package.config import config
from src.package.qt_dialogs import select_file


def main() -> int:
    """必要に応じて PyQt のファイル選択ダイアログを表示して処理を実行する。

    Returns
    -------
    int
        黒点候補の検出数。

    Raises
    ------
    ValueError
        ファイル選択がキャンセルされた場合。
    """
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

    return run_processing(config=config)


if __name__ == "__main__":
    main()
