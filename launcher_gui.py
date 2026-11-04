"""Windows batch launcher GUI for selecting application and environment options."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BATCH_FILE = PROJECT_ROOT / "run.bat"
SOURCE_INPUT_DIR = PROJECT_ROOT / "source_input"

ACTIONS = (
    ("アプリケーションを起動", "app"),
    ("仮想環境を構築・更新 (--setup)", "setup_venv"),
    ("システムPythonにインストール (--setup --global)", "setup_global"),
    ("ヘルプを表示 (--help)", "help"),
)


def build_batch_arguments(
    action: str,
    *,
    use_venv: bool = False,
    compare: bool = False,
    date_folder: str = "",
) -> list[str]:
    """Build the existing run.bat arguments selected in the GUI."""
    if action == "app":
        arguments: list[str] = []
        if use_venv:
            arguments.append("--venv")
        if compare:
            arguments.append("--compare")
            folder = date_folder.strip()
            if folder:
                arguments.append(folder)
        return arguments
    if action == "setup_venv":
        return ["--setup"]
    if action == "setup_global":
        return ["--setup", "--global"]
    if action == "help":
        return ["--help"]
    raise ValueError(f"Unknown launcher action: {action}")


def _run_batch(arguments: list[str]) -> int:
    """Run run.bat in the current console with the GUI bypass marker."""
    command = subprocess.list2cmdline(
        [str(BATCH_FILE), "--_from-gui", *arguments]
    )
    completed = subprocess.run(
        [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/s",
            "/c",
            f'"{command}"',
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return completed.returncode


def main() -> int:
    """Display the launcher GUI and run the selected batch operation."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as error:
        print(
            "[ERROR] Tkinter is required to display the run.bat launcher GUI. "
            "Install or select a Python 3.13+ distribution that includes Tcl/Tk."
        )
        print(f"Details: {error}")
        return 1

    root = tk.Tk()
    root.title("太陽黒点検出 - 起動メニュー")
    root.resizable(False, False)

    action_var = tk.StringVar(value="app")
    use_venv_var = tk.BooleanVar(value=False)
    compare_var = tk.BooleanVar(value=False)
    folder_var = tk.StringVar(value="")

    outer = ttk.Frame(root, padding=16)
    outer.grid(row=0, column=0, sticky="nsew")
    outer.columnconfigure(0, weight=1)

    action_frame = ttk.LabelFrame(outer, text="起動する処理", padding=10)
    action_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    for index, (label, value) in enumerate(ACTIONS):
        ttk.Radiobutton(
            action_frame,
            text=label,
            variable=action_var,
            value=value,
            command=lambda: update_controls(),
        ).grid(
            row=index // 2,
            column=index % 2,
            sticky="w",
            padx=(0, 18),
            pady=3,
        )

    environment_frame = ttk.LabelFrame(
        outer,
        text="Python実行環境",
        padding=10,
    )
    environment_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    venv_check = ttk.Checkbutton(
        environment_frame,
        text="既存の .venv を強制使用する (--venv)",
        variable=use_venv_var,
    )
    venv_check.grid(row=0, column=0, sticky="w")
    ttk.Label(
        environment_frame,
        text="未選択の場合は、既存の .venv を優先し、存在しなければシステムPythonを使用します。",
        wraplength=620,
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    application_frame = ttk.LabelFrame(
        outer,
        text="run.py に渡す引数",
        padding=10,
    )
    application_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    compare_check = ttk.Checkbutton(
        application_frame,
        text="比較モードを使用する (--compare)",
        variable=compare_var,
        command=lambda: update_controls(),
    )
    compare_check.grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(
        application_frame,
        text="日付フォルダー (任意)",
    ).grid(row=1, column=0, sticky="w", pady=(8, 0))
    folder_entry = ttk.Entry(
        application_frame,
        textvariable=folder_var,
        width=64,
    )
    folder_entry.grid(row=2, column=0, sticky="ew", pady=(3, 0))
    browse_button = ttk.Button(
        application_frame,
        text="参照...",
        command=lambda: choose_folder(),
    )
    browse_button.grid(row=2, column=1, sticky="e", padx=(8, 0), pady=(3, 0))
    application_frame.columnconfigure(0, weight=1)
    ttk.Label(
        application_frame,
        text="フォルダーを指定しない場合は、run.py のフォルダー選択ダイアログを表示します。",
        wraplength=620,
    ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

    button_frame = ttk.Frame(outer)
    button_frame.grid(row=3, column=0, sticky="e")
    exit_code = 0

    def choose_folder() -> None:
        """Choose the comparison date folder."""
        initial_directory = (
            SOURCE_INPUT_DIR if SOURCE_INPUT_DIR.is_dir() else PROJECT_ROOT
        )
        selected = filedialog.askdirectory(
            title="比較する日付フォルダーを選択してください",
            initialdir=str(initial_directory),
            mustexist=True,
        )
        if selected:
            folder_var.set(selected)

    def update_controls() -> None:
        """Enable only controls relevant to the selected operation."""
        app_selected = action_var.get() == "app"
        compare_selected = app_selected and compare_var.get()
        venv_check.configure(
            state="normal" if app_selected else "disabled"
        )
        compare_check.configure(
            state="normal" if app_selected else "disabled"
        )
        folder_entry.configure(
            state="normal" if compare_selected else "disabled"
        )
        browse_button.configure(
            state="normal" if compare_selected else "disabled"
        )

    def execute_selection() -> None:
        """Run the selected operation through the existing batch file."""
        nonlocal exit_code
        action = action_var.get()
        date_folder = folder_var.get().strip()
        if action == "app" and compare_var.get() and date_folder:
            if not Path(date_folder).expanduser().is_dir():
                messagebox.showerror(
                    "フォルダーが見つかりません",
                    "指定した日付フォルダーが存在しません。",
                    parent=root,
                )
                return

        arguments = build_batch_arguments(
            action,
            use_venv=use_venv_var.get(),
            compare=compare_var.get(),
            date_folder=date_folder,
        )
        root.withdraw()
        try:
            exit_code = _run_batch(arguments)
        except OSError as error:
            root.deiconify()
            messagebox.showerror(
                "起動エラー",
                f"run.bat を起動できませんでした。\n{error}",
                parent=root,
            )
            return
        root.destroy()

    ttk.Button(
        button_frame,
        text="実行",
        command=execute_selection,
    ).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(
        button_frame,
        text="キャンセル",
        command=root.destroy,
    ).grid(row=0, column=1)

    update_controls()
    root.update_idletasks()
    window_width = root.winfo_reqwidth()
    window_height = root.winfo_reqheight()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_x = max((screen_width - window_width) // 2, 0)
    position_y = max((screen_height - window_height) // 2, 0)
    root.geometry(
        f"{window_width}x{window_height}+{position_x}+{position_y}"
    )
    root.mainloop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
