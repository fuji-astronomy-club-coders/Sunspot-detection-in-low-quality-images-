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
    """PyQtダイアログで複数の矩形マスク領域を選択する。

    ドラッグで矩形を追加し、Enterで確定、Backspaceで取り消し、
    Escで確定せずに終了する。説明テキストはQPainterでベクター描画する。
    """
    from PyQt6.QtCore import Qt, QRectF, QPointF
    from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
    from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QWidget

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
    # OpenCVのBGR画像をQtが扱うRGB画像に変換
    rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    qimage = QImage(
        rgb_image.data, image_width, image_height, int(rgb_image.strides[0]),
        QImage.Format.Format_RGB888,
    ).copy()
    pixmap = QPixmap.fromImage(qimage)
    regions: list[tuple[int, int, int, int]] = []
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])

    class ImageCanvas(QWidget):
        """画像とROI、ベクターテキストを描画するキャンバス。"""

        def __init__(self) -> None:
            super().__init__()
            self.setMinimumSize(320, 240)
            self.drawing = False
            self.start = QPointF()
            self.current = QPointF()
            self.mouse = QPointF(-1, -1)

        def image_rect(self) -> QRectF:
            scale = min(self.width() / image_width, self.height() / image_height)
            w, h = image_width * scale, image_height * scale
            return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

        def to_image(self, point: QPointF) -> tuple[int, int]:
            rect = self.image_rect()
            x = round((point.x() - rect.x()) * image_width / rect.width())
            y = round((point.y() - rect.y()) * image_height / rect.height())
            return (min(max(x, 0), image_width - 1), min(max(y, 0), image_height - 1))

        def paintEvent(self, event) -> None:
            del event
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor(28, 28, 28))
            rect = self.image_rect()
            painter.drawPixmap(rect, pixmap, QRectF(pixmap.rect()))
            color_values = options["rectangle_color"]
            color = QColor(int(color_values[2]), int(color_values[1]), int(color_values[0]))
            pen = QPen(color, max(1, int(options["rectangle_thickness"])))
            painter.setPen(pen)

            def draw_roi(x: int, y: int, w: int, h: int) -> None:
                sx = rect.x() + x * rect.width() / image_width
                sy = rect.y() + y * rect.height() / image_height
                painter.drawRect(QRectF(sx, sy, w * rect.width() / image_width,
                                         h * rect.height() / image_height))

            for roi in regions:
                draw_roi(*roi)
            if self.drawing:
                x1, y1 = self.to_image(self.start)
                x2, y2 = self.to_image(self.current)
                draw_roi(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))

            lines = []
            if options["show_coordinates"]:
                if self.mouse.x() < 0:
                    lines.append(f"Cursor: ---, ---  |  Image: {image_width} x {image_height}")
                else:
                    mx, my = self.to_image(self.mouse)
                    lines.append(f"Cursor: x={mx}, y={my}  |  Image: {image_width} x {image_height}")
                if self.drawing:
                    x1, y1 = self.to_image(self.start)
                    x2, y2 = self.to_image(self.current)
                    lines.append(f"Selection: x={min(x1,x2)}, y={min(y1,y2)}, "
                                 f"width={abs(x2-x1)}, height={abs(y2-y1)}")
                elif regions:
                    x, y, w, h = regions[-1]
                    lines.append(f"Last ROI: x={x}, y={y}, width={w}, height={h}")
            if options["show_instructions"]:
                lines.append("Drag: select ROI | Enter: finish | Backspace: undo | Esc: cancel")
            if options["show_scale"]:
                lines.append(f"Display scale: {rect.width()/image_width*100:.1f}%")
            painter.setFont(self.font())
            metrics = painter.fontMetrics()
            y = 8 + metrics.ascent()
            for line in lines:
                # QPainterの文字描画はベクター描画で、拡大時も文字が劣化しにくい
                painter.setPen(QPen(QColor(0, 0, 0), 3))
                painter.drawText(13, y + 1, line)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(12, y, line)
                y += metrics.height() + 5
            painter.end()

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton and self.image_rect().contains(event.position()):
                self.drawing = True
                self.start = event.position()
                self.current = event.position()
                self.update()

        def mouseMoveEvent(self, event) -> None:
            self.mouse = event.position()
            if self.drawing:
                self.current = event.position()
            self.update()

        def mouseReleaseEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton and self.drawing:
                self.current = event.position()
                x1, y1 = self.to_image(self.start)
                x2, y2 = self.to_image(self.current)
                left, top = min(x1, x2), min(y1, y2)
                width, height = abs(x2-x1), abs(y2-y1)
                if width > 0 and height > 0:
                    regions.append((left, top, width, height))
                self.drawing = False
                self.update()

    class MaskDialog(QDialog):
        """縦横比を維持するマスク選択ダイアログ。"""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Select Mask Regions")
            self._aspect = max(320, int(options["window_width"])) / max(240, int(options["window_height"]))
            self._resizing = False
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self.canvas = ImageCanvas()
            layout.addWidget(self.canvas)
            self.resize(max(320, int(options["window_width"])),
                        max(240, int(options["window_height"])))
            if not options["resizable"]:
                self.setFixedSize(self.size())

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            if self._resizing or not options["resizable"]:
                return
            self._resizing = True
            size = event.size()
            # 変更された方向に合わせてもう一方を調整し、縦横比を維持する
            old = event.oldSize()
            if old.isValid() and abs(size.width() - old.width()) >= abs(size.height() - old.height()):
                self.resize(size.width(), max(240, round(size.width() / self._aspect)))
            else:
                self.resize(max(320, round(size.height() * self._aspect)), size.height())
            self._resizing = False

        def keyPressEvent(self, event) -> None:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.accept()
            elif event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                if regions:
                    regions.pop()
                    self.canvas.update()
            elif event.key() == Qt.Key.Key_Escape:
                self.reject()
            else:
                super().keyPressEvent(event)

    dialog = MaskDialog()
    result = dialog.exec()
    if result != QDialog.DialogCode.Accepted:
        regions.clear()
    dialog.deleteLater()
    if owns_app:
        app.quit()

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
