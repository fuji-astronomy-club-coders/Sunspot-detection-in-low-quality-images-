"""Run the application"""

from tkinter.filedialog import askopenfilename

from src.main import main
from src.package.config import config

if config["Paths"]["input_image"] == "?Dialog?":
    # ファイル選択ダイアログを表示して、ユーザーに画像ファイルを選択させる
    selected_file = askopenfilename(title="Select an image file", filetypes=[("Image files", "*.tiff *.tif *.fits *.fit *.fts")])
    if selected_file:
        config["Paths"]["input_image"] = selected_file
    else:
        raise ValueError("No image file selected. Please select an image file to proceed.")
main(config=config)
