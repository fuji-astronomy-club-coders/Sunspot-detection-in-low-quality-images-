"""PyQt ベースの共通ファイル選択ダイアログを提供するモジュール。"""

import sys

from PyQt6.QtWidgets import QApplication, QFileDialog

_QT_APPLICATION: QApplication | None = None


def ensure_qapplication() -> QApplication:
    """既存の QApplication を取得し、なければ生成する。

    Returns
    -------
    QApplication
        ダイアログ表示に使用する Qt アプリケーション。
    """
    global _QT_APPLICATION

    application = QApplication.instance()
    if isinstance(application, QApplication):
        app = application
    elif application is None:
        app = QApplication(sys.argv[:1])
        app.setQuitOnLastWindowClosed(False)
    else:
        raise RuntimeError(
            "QCoreApplication already exists; a QApplication is required for GUI dialogs."
        )

    _QT_APPLICATION = app
    return app


def select_file(
    title: str,
    name_filter: str,
    directory: str = "",
) -> str | None:
    """PyQt のファイル選択ダイアログでファイルを選択する。

    Parameters
    ----------
    title : str
        ダイアログのタイトル。
    name_filter : str
        QFileDialog に渡すファイル名フィルター。
    directory : str, optional
        初期表示するディレクトリまたはファイルパス。

    Returns
    -------
    str | None
        選択したファイルのパス。キャンセルされた場合は ``None``。
    """
    ensure_qapplication()
    selected_path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        directory,
        name_filter,
    )
    return selected_path or None


def select_directory(title: str, directory: str = "") -> str | None:
    """PyQt のフォルダー選択ダイアログでディレクトリを選択する。

    Parameters
    ----------
    title : str
        ダイアログのタイトル。
    directory : str, optional
        初期表示するディレクトリ。

    Returns
    -------
    str | None
        選択したディレクトリのパス。キャンセルされた場合は ``None``。
    """
    ensure_qapplication()
    selected_path = QFileDialog.getExistingDirectory(
        None,
        title,
        directory,
    )
    return selected_path or None
