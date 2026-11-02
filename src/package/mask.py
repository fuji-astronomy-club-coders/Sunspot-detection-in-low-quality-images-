"""黒点候補を矩形領域によって除外するモジュール。"""

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QCloseEvent,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QResizeEvent,
)
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.package.qt_dialogs import ensure_qapplication


def _to_display_image(image: np.ndarray, colormap: str) -> np.ndarray:
    """入力画像を指定カラーマップの表示用 BGR 画像へ変換する。

    Parameters
    ----------
    image : numpy.ndarray
        表示対象の2次元画像。
    colormap : str
        OpenCV のカラーマップ名。``GRAY`` の場合はグレースケール表示。

    Returns
    -------
    numpy.ndarray
        8 bit の BGR 表示画像。

    Raises
    ------
    ValueError
        未対応のカラーマップが指定された場合。
    """
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


class _MaskCanvas(QWidget):
    """画像とマスク矩形を QPainter で描画するキャンバス。"""

    regionsChanged = pyqtSignal(int)

    def __init__(
        self,
        image: np.ndarray,
        display_image: np.ndarray,
        options: dict,
        parent: QWidget | None = None,
    ) -> None:
        """キャンバスと画像表示用の QImage を初期化する。

        Parameters
        ----------
        image : numpy.ndarray
            元画像。マウス座標を画像座標へ変換するために使用する。
        display_image : numpy.ndarray
            カラーマップ適用後の BGR 表示画像。
        options : dict
            座標、説明、倍率、矩形描画に関する設定。
        parent : QWidget | None, optional
            親ウィジェット。
        """
        super().__init__(parent)
        self.image_height, self.image_width = image.shape
        rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        self._qimage = QImage(
            rgb_image.data,
            self.image_width,
            self.image_height,
            int(rgb_image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._options = options
        self.regions: list[tuple[int, int, int, int]] = []
        self._drawing = False
        self._start_point = (0, 0)
        self._current_point = (0, 0)
        self._mouse_point: tuple[int, int] | None = None
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _image_rect(self) -> QRectF:
        """キャンバス内で画像をアスペクト比を保って配置する領域を返す。

        Returns
        -------
        QRectF
            画像の描画先領域。
        """
        scale = min(
            self.width() / self.image_width,
            self.height() / self.image_height,
        )
        shown_width = self.image_width * scale
        shown_height = self.image_height * scale
        return QRectF(
            (self.width() - shown_width) / 2,
            (self.height() - shown_height) / 2,
            shown_width,
            shown_height,
        )

    def _map_to_image(
        self,
        position: QPointF,
        clamp: bool = False,
    ) -> tuple[int, int] | None:
        """ウィジェット座標を元画像の画素座標へ変換する。

        Parameters
        ----------
        position : QPointF
            キャンバス上の座標。
        clamp : bool, optional
            True の場合、画像外の座標を画像端へ丸める。

        Returns
        -------
        tuple[int, int] | None
            画像上の ``(x, y)``。画像外で ``clamp=False`` の場合は ``None``。
        """
        target = self._image_rect()
        if not clamp and not target.contains(position):
            return None
        x = (position.x() - target.left()) / target.width() * self.image_width
        y = (position.y() - target.top()) / target.height() * self.image_height
        return (
            min(max(int(x), 0), self.image_width - 1),
            min(max(int(y), 0), self.image_height - 1),
        )

    def _overlay_lines(self, scale: float) -> list[str]:
        """設定に応じた座標、選択状態、操作説明を生成する。

        Parameters
        ----------
        scale : float
            画像の画面表示倍率。

        Returns
        -------
        list[str]
            キャンバスへベクター描画するテキスト行。
        """
        lines: list[str] = []
        if self._options["show_coordinates"]:
            if self._mouse_point is None:
                lines.append(
                    f"Cursor: ---, ---  |  Image: {self.image_width} x {self.image_height}"
                )
            else:
                lines.append(
                    f"Cursor: x={self._mouse_point[0]}, y={self._mouse_point[1]}  "
                    f"|  Image: {self.image_width} x {self.image_height}"
                )
            if self._drawing:
                x1, y1 = self._start_point
                x2, y2 = self._current_point
                lines.append(
                    f"Selection: x={min(x1, x2)}, y={min(y1, y2)}, "
                    f"width={abs(x2 - x1)}, height={abs(y2 - y1)}"
                )
            elif self.regions:
                x, y, width, height = self.regions[-1]
                lines.append(f"Last ROI: x={x}, y={y}, width={width}, height={height}")

        if self._options["show_instructions"]:
            lines.append("Drag: select ROI | Enter/Esc: finish | Backspace: undo")
        if self._options["show_scale"]:
            lines.append(f"Display scale: {scale * 100:.1f}%")
        return lines

    def paintEvent(self, event: QPaintEvent) -> None:
        """画像、矩形、座標情報を Qt のベクター描画で再描画する。

        Parameters
        ----------
        event : QPaintEvent
            Qt の描画イベント。
        """
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 28, 28))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        target = self._image_rect()
        painter.drawImage(target, self._qimage)

        # 画像の外周を二重線で囲み、暗い画像・明るい画像のどちらでも
        # 表示領域の端が分かるようにする。画像データ自体には描き込まない。
        image_border = target.adjusted(-0.5, -0.5, 0.5, 0.5)
        outer_border = QPen(QColor(16, 16, 16, 255), 5)
        outer_border.setCosmetic(True)
        painter.setPen(outer_border)
        painter.drawRect(image_border)
        inner_border = QPen(QColor(255, 220, 0, 255), 2)
        inner_border.setCosmetic(True)
        painter.setPen(inner_border)
        painter.drawRect(image_border)

        scale_x = target.width() / self.image_width
        scale_y = target.height() / self.image_height

        color_values = [int(value) for value in self._options["rectangle_color"]]
        if len(color_values) != 3:
            color_values = [0, 255, 0]
        color = QColor(*reversed(color_values))
        pen = QPen(color, float(self._options["rectangle_thickness"]))
        pen.setCosmetic(True)
        painter.setPen(pen)

        for x, y, width, height in self.regions:
            painter.drawRect(
                QRectF(
                    target.left() + x * scale_x,
                    target.top() + y * scale_y,
                    width * scale_x,
                    height * scale_y,
                )
            )

        if self._drawing:
            x1, y1 = self._start_point
            x2, y2 = self._current_point
            painter.drawRect(
                QRectF(
                    target.left() + min(x1, x2) * scale_x,
                    target.top() + min(y1, y2) * scale_y,
                    abs(x2 - x1) * scale_x,
                    abs(y2 - y1) * scale_y,
                )
            )

        overlay_lines = self._overlay_lines(min(scale_x, scale_y))
        if overlay_lines:
            font = QFont("Meiryo", 10)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            line_height = metrics.height() + 4
            text_width = max(metrics.horizontalAdvance(line) for line in overlay_lines)
            box = QRectF(
                target.left() + 10,
                target.top() + 10,
                text_width + 20,
                line_height * len(overlay_lines) + 12,
            )
            painter.fillRect(box, QColor(0, 0, 0, 165))
            painter.setPen(QColor(255, 255, 255))
            for index, line in enumerate(overlay_lines):
                painter.drawText(
                    QPointF(box.left() + 10, box.top() + 6 + metrics.ascent() + index * line_height),
                    line,
                )
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """左クリック位置から矩形選択を開始する。

        Parameters
        ----------
        event : QMouseEvent
            マウス押下イベント。
        """
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._map_to_image(event.position())
        if point is None:
            return
        self.setFocus()
        self._drawing = True
        self._start_point = point
        self._current_point = point
        self._mouse_point = point
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """カーソル座標とドラッグ中の矩形終点を更新する。

        Parameters
        ----------
        event : QMouseEvent
            マウス移動イベント。
        """
        point = self._map_to_image(event.position(), clamp=self._drawing)
        self._mouse_point = point
        if self._drawing and point is not None:
            self._current_point = point
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """ドラッグを終了し、有効な矩形を選択領域へ追加する。

        Parameters
        ----------
        event : QMouseEvent
            マウス解放イベント。
        """
        if event.button() != Qt.MouseButton.LeftButton or not self._drawing:
            return
        point = self._map_to_image(event.position(), clamp=True)
        if point is not None:
            self._current_point = point
            x1, y1 = self._start_point
            x2, y2 = self._current_point
            left, top = min(x1, x2), min(y1, y2)
            width, height = abs(x2 - x1), abs(y2 - y1)
            if width > 0 and height > 0:
                self.regions.append((left, top, width, height))
                self.regionsChanged.emit(len(self.regions))
        self._drawing = False
        self.update()

    def undo(self) -> None:
        """最後に追加した矩形を取り消す。

        Returns
        -------
        None
        """
        if self.regions:
            self.regions.pop()
            self.regionsChanged.emit(len(self.regions))
        self.update()


class _MaskDialog(QDialog):
    """縦横比を維持するマスク領域選択ダイアログ。"""

    def __init__(
        self,
        image: np.ndarray,
        options: dict,
        parent: QWidget | None = None,
    ) -> None:
        """画像キャンバスと操作ボタンを持つダイアログを構築する。

        Parameters
        ----------
        image : numpy.ndarray
            マスク領域を選択する2次元画像。
        options : dict
            表示設定および初期ウィンドウサイズ。
        parent : QWidget | None, optional
            親ウィジェット。
        """
        super().__init__(parent)
        self._resizable = bool(options["resizable"])
        self._aspect_ratio = max(
            0.1,
            float(options["window_width"]) / max(float(options["window_height"]), 1.0),
        )
        self._adjusting_size = False
        self.setWindowTitle("Select Mask Regions")

        self.canvas = _MaskCanvas(image, _to_display_image(image, str(options["colormap"])), options)
        self.region_count_label = QLabel("選択領域: 0")
        undo_button = QPushButton("1つ戻す")
        undo_button.clicked.connect(self.canvas.undo)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button is not None:
            ok_button.setText("確定")
        if cancel_button is not None:
            cancel_button.setText("キャンセル")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.canvas.regionsChanged.connect(
            lambda count: self.region_count_label.setText(f"選択領域: {count}")
        )

        footer = QHBoxLayout()
        footer.addWidget(self.region_count_label)
        footer.addStretch(1)
        footer.addWidget(undo_button)
        footer.addWidget(button_box)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(footer)

        requested_width = max(480, int(options["window_width"]))
        requested_height = max(360, int(options["window_height"]))
        initial_width = max(requested_width, round(requested_height * self._aspect_ratio))
        initial_height = round(initial_width / self._aspect_ratio)
        if self._resizable:
            minimum_width = 640
            minimum_height = round(minimum_width / self._aspect_ratio)
            self.setMinimumSize(minimum_width, minimum_height)
            self.resize(initial_width, initial_height)
        else:
            self.setFixedSize(initial_width, initial_height)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """ユーザーによるリサイズ時にウィンドウの縦横比を維持する。

        Parameters
        ----------
        event : QResizeEvent
            Qt のリサイズイベント。
        """
        super().resizeEvent(event)
        if not self._resizable or self._adjusting_size:
            return
        size = event.size()
        old_size = event.oldSize()
        if not old_size.isValid():
            return

        delta_width = abs(size.width() - old_size.width())
        delta_height = abs(size.height() - old_size.height())
        if delta_width >= delta_height:
            target_width = size.width()
            target_height = round(target_width / self._aspect_ratio)
        else:
            target_height = size.height()
            target_width = round(target_height * self._aspect_ratio)

        if target_width < self.minimumWidth():
            target_width = self.minimumWidth()
            target_height = round(target_width / self._aspect_ratio)
        if target_height < self.minimumHeight():
            target_height = self.minimumHeight()
            target_width = round(target_height * self._aspect_ratio)
        if (target_width, target_height) != (size.width(), size.height()):
            self._adjusting_size = True
            self.resize(target_width, target_height)
            self._adjusting_size = False

    def closeEvent(self, event: QCloseEvent) -> None:
        """ウィンドウを閉じた場合も現在の矩形選択を確定する。

        Parameters
        ----------
        event : QCloseEvent
            Qt のウィンドウ終了イベント。

        Returns
        -------
        None
        """
        self.done(QDialog.DialogCode.Accepted)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Enter、Backspace、Esc キーの操作を処理する。

        Parameters
        ----------
        event : QKeyEvent
            キー押下イベント。
        """
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
        elif event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.canvas.undo()
            self.region_count_label.setText(f"選択領域: {len(self.canvas.regions)}")
        elif event.key() == Qt.Key.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)


def selectMaskRegions(
    image: np.ndarray,
    display_options: dict | None = None,
) -> list[tuple[int, int, int, int]]:
    """PyQt ダイアログで複数の矩形マスク領域を選択する。

    画像上をドラッグして矩形を作成する。Enter、Esc、または「確定」で確定し、
    Backspace / Delete で直前の矩形を取り消す。「キャンセル」では選択を破棄する。

    Parameters
    ----------
    image : numpy.ndarray
        選択時に表示する2次元画像。
    display_options : dict | None, optional
        ``config.yaml`` の ``MaskDialog.display`` に対応する表示設定。

    Returns
    -------
    list[tuple[int, int, int, int]]
        確定した矩形領域。各要素は ``(x, y, width, height)``。

    Raises
    ------
    ValueError
        入力画像が2次元ではない場合。
    """
    if image.ndim != 2:
        raise ValueError("入力画像はグレースケール画像にしてください。")
    if image.size == 0:
        raise ValueError("入力画像が空です。")

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

    ensure_qapplication()
    dialog = _MaskDialog(image, options)
    regions = (
        dialog.canvas.regions.copy()
        if dialog.exec() == QDialog.DialogCode.Accepted
        else []
    )

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
    labels : numpy.ndarray
        黒点ごとに異なる整数ラベルを持つ画像。
    regions : list[tuple[int, int, int, int]]
        除外する矩形領域のリスト。各要素は ``(x, y, width, height)``。
    original_count : int
        マスク適用前の黒点候補数。

    Returns
    -------
    tuple[int, numpy.ndarray]
        マスク適用後の候補数と二値画像。

    Raises
    ------
    ValueError
        ラベル画像が2次元ではない場合。
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
