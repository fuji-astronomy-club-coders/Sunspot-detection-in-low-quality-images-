"""黒点候補を矩形領域によって除外するモジュール。"""

import cv2
import numpy as np


def _to_display_image(image: np.ndarray, colormap: str) -> np.ndarray:
    """入力画像を指定カラーマップの表示用BGR画像へ変換する。"""
    normalized = np.empty(image.shape, dtype=np.uint8)
    cv2.normalize(
        image,
        normalized,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    colormaps = {
        "AUTUMN": cv2.COLORMAP_AUTUMN,
        "BONE": cv2.COLORMAP_BONE,
        "JET": cv2.COLORMAP_JET,
        "WINTER": cv2.COLORMAP_WINTER,
        "RAINBOW": cv2.COLORMAP_RAINBOW,
        "OCEAN": cv2.COLORMAP_OCEAN,
        "SUMMER": cv2.COLORMAP_SUMMER,
        "SPRING": cv2.COLORMAP_SPRING,
        "COOL": cv2.COLORMAP_COOL,
        "HSV": cv2.COLORMAP_HSV,
        "PINK": cv2.COLORMAP_PINK,
        "HOT": cv2.COLORMAP_HOT,
        "PARULA": cv2.COLORMAP_PARULA,
        "TURBO": cv2.COLORMAP_TURBO,
        "TWILIGHT": cv2.COLORMAP_TWILIGHT,
        "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "CIVIDIS": cv2.COLORMAP_CIVIDIS,
    }
    if colormap.upper() == "GRAY":
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    if colormap.upper() not in colormaps:
        raise ValueError(f"未対応のカラーマップです: {colormap}")
    return cv2.applyColorMap(normalized, colormaps[colormap.upper()])


def selectMaskRegions(
    image: np.ndarray,
    display_options: dict | None = None,
) -> list[tuple[int, int, int, int]]:
    """画像上で複数の矩形マスク領域を対話的に選択する。

    マウスドラッグで矩形を作成する。Enterで確定、Backspaceで直前の矩形を
    取り消し、Escで選択を終了する。

    Parameters
    ----------
    image : np.ndarray
        選択時に表示する2次元画像。
    display_options : dict | None
        config.yaml の ``MaskDialog.display`` に対応する表示設定。

    Returns
    -------
    list[tuple[int, int, int, int]]
        選択した矩形領域。各要素は (x, y, width, height)。
    """
    if image.ndim != 2:
        raise ValueError("入力画像はグレースケール画像にしてください。")

    options = {
        "colormap": "HOT",
        "show_coordinates": True,
        "show_instructions": True,
        "resizable": True,
        "show_scale": True,
        "window_width": 1100,
        "window_height": 800,
        "rectangle_color": [0, 255, 0],
        "rectangle_thickness": 2,
    }
    if display_options:
        options.update(display_options)

    display_image = _to_display_image(image, str(options["colormap"]))
    image_height, image_width = image.shape
    regions: list[tuple[int, int, int, int]] = []
    window_name = "Select Mask Regions"

    drawing = False
    start_point = (0, 0)
    current_point = (0, 0)
    mouse_point: tuple[int, int] | None = None
    window_width = max(320, int(options["window_width"]))
    window_height = max(240, int(options["window_height"]))
    resizable = bool(options["resizable"])

    def clamp_point(x: int, y: int) -> tuple[int, int]:
        return (
            min(max(int(x), 0), image_width - 1),
            min(max(int(y), 0), image_height - 1),
        )

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal drawing, start_point, current_point, mouse_point
        del flags, param
        point = clamp_point(x, y)
        mouse_point = point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = point
            current_point = point
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_point = point
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            current_point = point
            drawing = False
            x1, y1 = start_point
            x2, y2 = current_point
            left, top = min(x1, x2), min(y1, y2)
            width, height = abs(x2 - x1), abs(y2 - y1)
            if width > 0 and height > 0:
                regions.append((left, top, width, height))

    flags = cv2.WINDOW_NORMAL if resizable else cv2.WINDOW_AUTOSIZE
    cv2.namedWindow(window_name, flags)
    if resizable:
        cv2.resizeWindow(window_name, window_width, window_height)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            canvas = display_image.copy()
            color = tuple(int(v) for v in options["rectangle_color"])
            for x, y, width, height in regions:
                cv2.rectangle(
                    canvas,
                    (x, y),
                    (x + width, y + height),
                    color,
                    int(options["rectangle_thickness"]),
                )

            if drawing:
                cv2.rectangle(canvas, start_point, current_point, color, 1)

            lines: list[str] = []
            if options["show_coordinates"]:
                if mouse_point is None:
                    lines.append(f"Cursor: ---, ---  |  Image: {image_width} x {image_height}")
                else:
                    lines.append(
                        f"Cursor: x={mouse_point[0]}, y={mouse_point[1]}  "
                        f"|  Image: {image_width} x {image_height}"
                    )
                if drawing:
                    x1, y1 = start_point
                    x2, y2 = current_point
                    lines.append(
                        f"Selection: x={min(x1, x2)}, y={min(y1, y2)}, "
                        f"width={abs(x2-x1)}, height={abs(y2-y1)}"
                    )
                elif regions:
                    x, y, width, height = regions[-1]
                    lines.append(
                        f"Last ROI: x={x}, y={y}, width={width}, height={height}"
                    )

            if options["show_instructions"]:
                lines.append("Drag: select ROI | Enter: finish | Backspace: undo | Esc: cancel/finish")

            if options["show_scale"]:
                try:
                    _, _, shown_width, shown_height = cv2.getWindowImageRect(window_name)
                    scale_x = shown_width / image_width if shown_width > 0 else 1.0
                    scale_y = shown_height / image_height if shown_height > 0 else 1.0
                    lines.append(
                        f"Display scale: {scale_x * 100:.1f}% x {scale_y * 100:.1f}%"
                    )
                except cv2.error:
                    lines.append(f"Image size: {image_width} x {image_height}")

            for index, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (12, 26 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    line,
                    (12, 26 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13):  # Enter
                break
            if key in (8, 127):  # Backspace / Delete
                if regions:
                    regions.pop()
            elif key == 27:  # Esc
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyWindow(window_name)

    print(f"合計 {len(regions)} 個のマスク領域を選択しました。")
    for index, (x, y, width, height) in enumerate(regions, start=1):
        print(f"マスク領域 {index}: x={x}, y={y}, width={width}, height={height}")
    return regions


def applyMask(
    labels: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    original_count: int,
) -> tuple[int, np.ndarray]:
    """矩形領域と重なる黒点候補を除外する。

    Parameters
    ----------
    labels : np.ndarray
        黒点ごとに異なる整数ラベルを持つ画像。
    regions : list[tuple[int, int, int, int]]
        (x, y, width, height) のリスト。
    original_count : int
        マスク適用前の黒点候補数。

    Returns
    -------
    tuple[int, np.ndarray]
        マスク適用後の候補数と二値画像。
    """
    if labels.ndim != 2:
        raise ValueError("ラベル画像は2次元配列にしてください。")

    masked_labels = labels.copy()
    height, width = labels.shape
    excluded_labels: set[int] = set()

    for x, y, roi_width, roi_height in regions:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + roi_width)
        y2 = min(height, y + roi_height)
        if x1 >= x2 or y1 >= y2:
            continue
        overlapping_labels = np.unique(labels[y1:y2, x1:x2])
        excluded_labels.update(
            int(label) for label in overlapping_labels if label != 0
        )

    if excluded_labels:
        masked_labels[np.isin(labels, list(excluded_labels))] = 0

    count = original_count - len(excluded_labels)
    print("除外した黒点数:", len(excluded_labels))
    print("マスク適用後の黒点数:", count)

    masked_image = np.where(masked_labels > 0, 0, 255).astype(np.uint8)
    return count, masked_image
