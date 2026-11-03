"""PyQt を使用した最終解析画像の確認ダイアログ。"""

import logging
import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QImage,
    QKeyEvent,
    QLinearGradient,
    QPainterPath,
    QKeySequence,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.package.qt_dialogs import ensure_qapplication

LOGGER = logging.getLogger("sunspot.qt_preview")


class _FinalPreviewCanvas(QWidget):
    """最終画像を表示し、ズーム・パン操作を受け付けるキャンバス。"""

    cursorChanged = pyqtSignal(str)
    viewChanged = pyqtSignal(str)
    historyChanged = pyqtSignal(bool, bool)

    def __init__(
        self,
        image: np.ndarray,
        mask_regions: list[tuple[int, int, int, int]] | None = None,
        display_options: dict | None = None,
        invalid_sunspot_mask: np.ndarray | None = None,
        solar_geometry: tuple[tuple[float, float], float] | None = None,
        rmask_radius_ratio: float = 0.85,
        parent: QWidget | None = None,
    ) -> None:
        """表示画像、マスク領域オーバーレイ、初期表示範囲を構築する。

        Parameters
        ----------
        image : numpy.ndarray
            最終確認する2次元画像。
        mask_regions : list[tuple[int, int, int, int]] | None, optional
            左上原点のマスク矩形。各要素は (x, y, width, height)。
        display_options : dict | None, optional
            最終確認画面のマスク塗りつぶし設定。
        invalid_sunspot_mask : numpy.ndarray | None, optional
            マスクで除外された黒点画素を True とした2次元配列。
        parent : QWidget | None, optional
            親ウィジェット。
        """
        super().__init__(parent)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("最終確認画像は空でない2次元配列にしてください。")

        options = display_options or {}
        self._mask_regions = [
            (int(x), int(y), int(width), int(height))
            for x, y, width, height in (mask_regions or [])
        ]
        self._show_mask_fill = bool(options.get("show_mask_fill", False))
        raw_color = options.get("mask_fill_color", [0, 255, 0])
        if not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
            raw_color = [0, 255, 0]
        self._mask_fill_color = (
            min(max(int(raw_color[0]), 0), 255),
            min(max(int(raw_color[1]), 0), 255),
            min(max(int(raw_color[2]), 0), 255),
        )
        self._mask_fill_alpha = min(
            max(int(options.get("mask_fill_alpha", 72)), 0),
            255,
        )
        self._show_rmask_fill = bool(options.get("show_rmask_fill", False))
        rmask_color = options.get("rmask_fill_color", [255, 165, 0])
        if not isinstance(rmask_color, (list, tuple)) or len(rmask_color) != 3:
            rmask_color = [255, 165, 0]
        self._rmask_fill_color = tuple(
            min(max(int(value), 0), 255) for value in rmask_color
        )
        self._rmask_fill_alpha = min(
            max(int(options.get("rmask_fill_alpha", 64)), 0),
            255,
        )
        self._show_solar_circle = bool(options.get("show_solar_circle", False))
        solar_color = options.get("solar_circle_color", [0, 220, 255])
        if not isinstance(solar_color, (list, tuple)) or len(solar_color) != 3:
            solar_color = [0, 220, 255]
        self._solar_circle_color = tuple(
            min(max(int(value), 0), 255) for value in solar_color
        )
        self._solar_circle_thickness = max(
            1.0,
            float(options.get("solar_circle_thickness", 2)),
        )
        self._solar_geometry: tuple[float, float, float] | None = None
        if solar_geometry is not None:
            (center_x, center_y), radius = solar_geometry
            geometry_values = (float(center_x), float(center_y), float(radius))
            if (
                not all(math.isfinite(value) for value in geometry_values)
                or geometry_values[2] <= 0
            ):
                raise ValueError("太陽近似円の中心・半径は有限値かつ正の半径である必要があります。")
            self._solar_geometry = geometry_values
        if not math.isfinite(rmask_radius_ratio) or rmask_radius_ratio <= 0:
            raise ValueError("Rmask.radius_ratio は正の有限値にしてください。")
        self._rmask_radius_ratio = float(rmask_radius_ratio)
        self._show_invalid_sunspots = bool(
            options.get("show_invalid_sunspots", True)
        )
        self._invalid_sunspots_qimage: QImage | None = None
        if invalid_sunspot_mask is not None:
            if invalid_sunspot_mask.ndim != 2 or invalid_sunspot_mask.shape != image.shape:
                raise ValueError("無効黒点マスクは表示画像と同じ形状の2次元配列にしてください。")
            invalid_overlay = np.zeros((*image.shape, 4), dtype=np.uint8)
            invalid_overlay[invalid_sunspot_mask.astype(bool)] = (255, 64, 64, 190)
            self._invalid_sunspots_qimage = QImage(
                invalid_overlay.tobytes(),
                image.shape[1],
                image.shape[0],
                int(invalid_overlay.strides[0]),
                QImage.Format.Format_RGBA8888,
            ).copy()

        values = np.nan_to_num(
            image.astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._minimum = float(values.min())
        self._maximum = float(values.max())
        if self._maximum > self._minimum:
            normalized = np.rint(
                (values - self._minimum)
                / (self._maximum - self._minimum)
                * 255
            ).astype(np.uint8)
        else:
            normalized = np.full(values.shape, 127, dtype=np.uint8)

        # マスク選択画面と同じ左上原点の画像配列を、そのまま表示する。
        normalized = np.ascontiguousarray(normalized)
        self._image_height, self._image_width = normalized.shape
        image_bytes = normalized.tobytes()
        self._qimage = QImage(
            image_bytes,
            self._image_width,
            self._image_height,
            int(normalized.strides[0]),
            QImage.Format.Format_Grayscale8,
        ).copy()

        self._view_x_min = 0.0
        self._view_x_max = float(self._image_width)
        self._view_y_min = 0.0
        self._view_y_max = float(self._image_height)
        self._view_history: list[tuple[float, float, float, float]] = [
            (0.0, float(self._image_width), 0.0, float(self._image_height))
        ]
        self._history_index = 0
        self._navigation_mode = "none"
        self._drag_mode = "none"
        self._drag_start = QPointF()
        self._drag_start_view = (0.0, 0.0, 0.0, 0.0)
        self._drag_start_scale = 1.0
        self._zoom_rect: QRectF | None = None
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.cursorChanged.emit("Cursor: x=—, y=—")
        self._emit_view_range()

    def set_mask_fill_visible(self, visible: bool) -> None:
        """最終確認画面のマスク範囲塗りつぶし表示を切り替える。

        Parameters
        ----------
        visible : bool
            True の場合はマスク範囲を半透明で塗りつぶす。
        """
        if self._show_mask_fill == visible:
            return
        self._show_mask_fill = visible
        self.update()

    def set_rmask_fill_visible(self, visible: bool) -> None:
        """Rmask の円外領域の塗りつぶし表示を切り替える。"""
        if self._show_rmask_fill == visible:
            return
        self._show_rmask_fill = visible
        self.update()

    def set_solar_circle_visible(self, visible: bool) -> None:
        """MIN2 の太陽近似円と中心マーカーの表示を切り替える。"""
        if self._show_solar_circle == visible:
            return
        self._show_solar_circle = visible
        self.update()

    def set_invalid_sunspots_visible(self, visible: bool) -> None:
        """無効黒点のオーバーレイ表示を切り替える。

        Parameters
        ----------
        visible : bool
            True の場合はマスクで除外された黒点を赤色で重ねて表示する。
        """
        if self._show_invalid_sunspots == visible:
            return
        self._show_invalid_sunspots = visible
        self.update()

    def _plot_area(self) -> QRectF:
        """軸ラベルとカラーバーの余白を除いた描画領域を返す。

        Returns
        -------
        QRectF
            キャンバス内のプロット領域。
        """
        return QRectF(
            72,
            18,
            max(1, self.width() - 174),
            max(1, self.height() - 76),
        )

    def _view_geometry(self) -> tuple[QRectF, float]:
        """現在の表示範囲を、縦横比を保った描画矩形へ変換する。

        Returns
        -------
        tuple[QRectF, float]
            表示矩形と、画像座標1ピクセルあたりの表示倍率。
        """
        area = self._plot_area()
        view_width = max(self._view_x_max - self._view_x_min, 1e-9)
        view_height = max(self._view_y_max - self._view_y_min, 1e-9)
        scale = min(area.width() / view_width, area.height() / view_height)
        shown_width = view_width * scale
        shown_height = view_height * scale
        target = QRectF(
            area.left() + (area.width() - shown_width) / 2,
            area.top() + (area.height() - shown_height) / 2,
            shown_width,
            shown_height,
        )
        return target, scale

    def _map_to_data(self, position: QPointF) -> tuple[float, float] | None:
        """キャンバス座標を画像と同じ左上原点の画像座標へ変換する。

        Parameters
        ----------
        position : QPointF
            キャンバス上の座標。

        Returns
        -------
        tuple[float, float] | None
            画像上の X・Y 座標。画像表示領域外では None。
        """
        target, scale = self._view_geometry()
        if not target.contains(position):
            return None
        x = self._view_x_min + (position.x() - target.left()) / scale
        y = self._view_y_min + (position.y() - target.top()) / scale
        return x, y

    @staticmethod
    def _clamp_range(minimum: float, maximum: float, image_size: int) -> tuple[float, float]:
        """表示範囲を画像境界内へ平行移動する。

        Parameters
        ----------
        minimum : float
            表示範囲の下限。
        maximum : float
            表示範囲の上限。
        image_size : int
            画像の幅または高さ。

        Returns
        -------
        tuple[float, float]
            画像境界内に収めた表示範囲。
        """
        span = maximum - minimum
        if span >= image_size:
            return 0.0, float(image_size)
        if minimum < 0:
            maximum -= minimum
            minimum = 0.0
        if maximum > image_size:
            minimum -= maximum - image_size
            maximum = float(image_size)
        return minimum, maximum

    def _set_view(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        record_history: bool = True,
    ) -> None:
        """新しい表示範囲を設定し、HUD とキャンバスを更新する。

        Parameters
        ----------
        x_min : float
            X 表示範囲の下限。
        x_max : float
            X 表示範囲の上限。
        y_min : float
            Y 表示範囲の下限。
        y_max : float
            Y 表示範囲の上限。
        record_history : bool, optional
            True の場合、表示範囲を Back / Forward の履歴へ追加する。
        """
        values = (x_min, x_max, y_min, y_max)
        if not all(math.isfinite(value) for value in values):
            return
        if x_max - x_min < 1e-3 or y_max - y_min < 1e-3:
            return

        self._view_x_min, self._view_x_max = self._clamp_range(
            x_min, x_max, self._image_width
        )
        self._view_y_min, self._view_y_max = self._clamp_range(
            y_min, y_max, self._image_height
        )
        self._emit_view_range()
        if record_history:
            self._record_view_history()
        self.update()

    def _emit_view_range(self) -> None:
        """現在の X・Y 表示範囲をステータス表示へ通知する。"""
        self.viewChanged.emit(
            f"X: {self._view_x_min:.1f} – {self._view_x_max:.1f}  |  "
            f"Y: {self._view_y_min:.1f} – {self._view_y_max:.1f}"
        )

    def _record_view_history(self) -> None:
        """現在の表示範囲を履歴へ追加し、進む／戻るボタンの状態を通知する。"""
        current = (
            self._view_x_min,
            self._view_x_max,
            self._view_y_min,
            self._view_y_max,
        )
        previous = self._view_history[self._history_index]
        if all(math.isclose(left, right, abs_tol=1e-6) for left, right in zip(current, previous, strict=True)):
            return
        self._view_history = self._view_history[: self._history_index + 1]
        self._view_history.append(current)
        if len(self._view_history) > 50:
            self._view_history.pop(0)
        self._history_index = len(self._view_history) - 1
        self._emit_history_state()

    def _emit_history_state(self) -> None:
        """履歴の前後に移動できるかをツールバーへ通知する。"""
        self.historyChanged.emit(self._history_index > 0, self._history_index < len(self._view_history) - 1)

    def go_back(self) -> None:
        """1つ前の表示範囲へ戻す。"""
        if self._history_index <= 0:
            return
        self._history_index -= 1
        self._set_view(*self._view_history[self._history_index], record_history=False)
        self._emit_history_state()

    def go_forward(self) -> None:
        """1つ後の表示範囲へ進む。"""
        if self._history_index >= len(self._view_history) - 1:
            return
        self._history_index += 1
        self._set_view(*self._view_history[self._history_index], record_history=False)
        self._emit_history_state()

    def reset_view(self) -> None:
        """画像全体が表示される初期範囲へ戻す。"""
        self._set_view(
            0.0,
            float(self._image_width),
            0.0,
            float(self._image_height),
        )
        self._drag_mode = "none"
        self._zoom_rect = None
        self.update()

    def set_navigation_mode(self, mode: str) -> None:
        """左ドラッグ操作を Pan、Zoom、通常状態のいずれかへ切り替える。

        Parameters
        ----------
        mode : str
            "none"、"pan"、"zoom" のいずれか。
        """
        if mode not in ("none", "pan", "zoom"):
            raise ValueError(f"未対応のナビゲーションモードです: {mode}")
        self._navigation_mode = mode
        if mode == "zoom":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def _zoom_at(self, factor: float, anchor: tuple[float, float]) -> None:
        """指定座標を中心として表示範囲を拡大・縮小する。

        Parameters
        ----------
        factor : float
            拡大率。1 より大きい値で拡大し、0 より大きく 1 未満の値で縮小する。
        anchor : tuple[float, float]
            ズームの中心となる画像座標。
        """
        if factor <= 0:
            return
        anchor_x, anchor_y = anchor
        self._set_view(
            anchor_x - (anchor_x - self._view_x_min) / factor,
            anchor_x + (self._view_x_max - anchor_x) / factor,
            anchor_y - (anchor_y - self._view_y_min) / factor,
            anchor_y + (self._view_y_max - anchor_y) / factor,
        )

    def paintEvent(self, event: QPaintEvent | None) -> None:
        """画像、軸、カラーバー、ズーム選択範囲を描画する。

        Parameters
        ----------
        event : QPaintEvent
            Qt の描画イベント。
        """
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        image_rect, scale = self._view_geometry()
        source_rect = QRectF(
            self._view_x_min,
            self._view_y_min,
            self._view_x_max - self._view_x_min,
            self._view_y_max - self._view_y_min,
        )
        painter.save()
        painter.setClipRect(self._plot_area())
        painter.drawImage(image_rect, self._qimage, source_rect)

        if self._show_mask_fill and self._mask_regions:
            fill_color = QColor(*self._mask_fill_color)
            fill_color.setAlpha(self._mask_fill_alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            for x, y, width, height in self._mask_regions:
                x_min = max(0, x)
                y_min = max(0, y)
                x_max = min(self._image_width, x + width)
                y_max = min(self._image_height, y + height)
                if x_min >= x_max or y_min >= y_max:
                    continue

                # Mask rectangles and the displayed image both use a top-left origin.
                rectangle = QRectF(
                    image_rect.left() + (x_min - self._view_x_min) * scale,
                    image_rect.top() + (y_min - self._view_y_min) * scale,
                    (x_max - x_min) * scale,
                    (y_max - y_min) * scale,
                )
                painter.drawRect(rectangle)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        if self._show_rmask_fill and self._solar_geometry is not None:
            center_x, center_y, solar_radius = self._solar_geometry
            radial_radius = solar_radius * self._rmask_radius_ratio
            center_screen = QPointF(
                image_rect.left() + (center_x - self._view_x_min) * scale,
                image_rect.top() + (center_y - self._view_y_min) * scale,
            )
            radial_circle = QRectF(
                center_screen.x() - radial_radius * scale,
                center_screen.y() - radial_radius * scale,
                2.0 * radial_radius * scale,
                2.0 * radial_radius * scale,
            )
            radial_path = QPainterPath()
            radial_path.setFillRule(Qt.FillRule.OddEvenFill)
            radial_path.addRect(image_rect)
            radial_path.addEllipse(radial_circle)
            radial_color = QColor(*self._rmask_fill_color)
            radial_color.setAlpha(self._rmask_fill_alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.fillPath(radial_path, radial_color)

        if self._show_invalid_sunspots and self._invalid_sunspots_qimage is not None:
            painter.drawImage(
                image_rect,
                self._invalid_sunspots_qimage,
                source_rect,
            )

        if self._show_solar_circle and self._solar_geometry is not None:
            center_x, center_y, solar_radius = self._solar_geometry
            center_screen = QPointF(
                image_rect.left() + (center_x - self._view_x_min) * scale,
                image_rect.top() + (center_y - self._view_y_min) * scale,
            )
            solar_circle = QRectF(
                center_screen.x() - solar_radius * scale,
                center_screen.y() - solar_radius * scale,
                2.0 * solar_radius * scale,
                2.0 * solar_radius * scale,
            )
            solar_pen = QPen(
                QColor(*self._solar_circle_color),
                self._solar_circle_thickness,
            )
            solar_pen.setCosmetic(True)
            painter.setPen(solar_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(solar_circle)
            cross_size = 8.0
            painter.drawLine(
                QPointF(center_screen.x() - cross_size, center_screen.y()),
                QPointF(center_screen.x() + cross_size, center_screen.y()),
            )
            painter.drawLine(
                QPointF(center_screen.x(), center_screen.y() - cross_size),
                QPointF(center_screen.x(), center_screen.y() + cross_size),
            )
        painter.restore()

        border_pen = QPen(QColor(235, 235, 235), 1)
        border_pen.setCosmetic(True)
        painter.setPen(border_pen)
        painter.drawRect(image_rect)

        painter.setPen(QColor(240, 240, 240))
        painter.drawText(
            QRectF(image_rect.left(), image_rect.bottom() + 8, image_rect.width(), 24),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "X [pixel]",
        )
        painter.save()
        painter.translate(image_rect.left() - 48, image_rect.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-image_rect.height() / 2, -12, image_rect.height(), 24),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Y [pixel]",
        )
        painter.restore()

        colorbar = QRectF(image_rect.right() + 18, image_rect.top(), 18, image_rect.height())
        gradient = QLinearGradient(
            colorbar.left(),
            colorbar.top(),
            colorbar.left(),
            colorbar.bottom(),
        )
        gradient.setColorAt(0.0, QColor(255, 255, 255))
        gradient.setColorAt(1.0, QColor(0, 0, 0))
        painter.fillRect(colorbar, gradient)
        painter.setPen(border_pen)
        painter.drawRect(colorbar)
        painter.drawText(
            QRectF(colorbar.right() + 7, colorbar.top() - 8, 78, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self._maximum:g}",
        )
        painter.drawText(
            QRectF(colorbar.right() + 7, colorbar.bottom() - 12, 78, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self._minimum:g}",
        )

        if self._zoom_rect is not None:
            zoom_pen = QPen(QColor(255, 220, 0), 1, Qt.PenStyle.DashLine)
            zoom_pen.setCosmetic(True)
            painter.setPen(zoom_pen)
            painter.setBrush(QColor(255, 220, 0, 35))
            painter.drawRect(self._zoom_rect.normalized())
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.end()

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        """Ctrl + ホイールでカーソル位置を中心に拡大・縮小する。

        Parameters
        ----------
        event : QWheelEvent
            Qt のホイールイベント。
        """
        if event is None:
            return
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            super().wheelEvent(event)
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        anchor = self._map_to_data(event.position())
        if anchor is None:
            return
        self._zoom_at(1.2**steps, anchor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Shift + 右ドラッグのパン、または選択モードの開始を処理する。

        Parameters
        ----------
        event : QMouseEvent
            Qt のマウス押下イベント。
        """
        if event is None:
            return
        if (
            event.button() == Qt.MouseButton.RightButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            if self._map_to_data(event.position()) is None:
                return
            self._drag_mode = "pan"
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._navigation_mode not in ("pan", "zoom"):
                return
            if self._map_to_data(event.position()) is None:
                return
            self._drag_mode = self._navigation_mode
        else:
            super().mousePressEvent(event)
            return

        self._drag_start = event.position()
        self._drag_start_view = (
            self._view_x_min,
            self._view_x_max,
            self._view_y_min,
            self._view_y_max,
        )
        _, self._drag_start_scale = self._view_geometry()
        if self._drag_mode == "zoom":
            self._zoom_rect = QRectF(self._drag_start, self._drag_start)
        else:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.setFocus()
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """カーソル座標を表示し、ドラッグ中のパン／ズーム矩形を更新する。

        Parameters
        ----------
        event : QMouseEvent
            Qt のマウス移動イベント。
        """
        if event is None:
            return
        position = event.position()
        point = self._map_to_data(position)
        if point is None:
            self.cursorChanged.emit("Cursor: x=—, y=—")
        else:
            pixel_x = min(max(int(point[0]), 0), self._image_width - 1)
            pixel_y = min(max(int(point[1]), 0), self._image_height - 1)
            self.cursorChanged.emit(f"Cursor: x={pixel_x}, y={pixel_y}")

        if self._drag_mode == "pan":
            x_min, x_max, y_min, y_max = self._drag_start_view
            delta_x = -(position.x() - self._drag_start.x()) / self._drag_start_scale
            delta_y = -(position.y() - self._drag_start.y()) / self._drag_start_scale
            self._set_view(
                x_min + delta_x,
                x_max + delta_x,
                y_min + delta_y,
                y_max + delta_y,
                record_history=False,
            )
        elif self._drag_mode == "zoom":
            self._zoom_rect = QRectF(self._drag_start, position).normalized()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        """ドラッグ終了時にパン状態を解除、または選択矩形へズームする。

        Parameters
        ----------
        event : QMouseEvent
            Qt のマウス解放イベント。
        """
        if event is None:
            return
        if (
            event.button() != Qt.MouseButton.LeftButton
            and event.button() != Qt.MouseButton.RightButton
        ):
            super().mouseReleaseEvent(event)
            return
        if self._drag_mode == "pan":
            self._record_view_history()
        if self._drag_mode == "zoom" and self._zoom_rect is not None:
            rectangle = self._zoom_rect.normalized()
            if rectangle.width() >= 4 and rectangle.height() >= 4:
                first = self._map_to_data(rectangle.topLeft())
                second = self._map_to_data(rectangle.bottomRight())
                if first is not None and second is not None:
                    self._set_view(
                        min(first[0], second[0]),
                        max(first[0], second[0]),
                        min(first[1], second[1]),
                        max(first[1], second[1]),
                    )
            self._zoom_rect = None
        self._drag_mode = "none"
        if self._navigation_mode == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._navigation_mode == "zoom":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        """Home キーで表示範囲を初期状態へ戻す。

        Parameters
        ----------
        event : QKeyEvent
            Qt のキー押下イベント。
        """
        if event is None:
            return
        if event.key() == Qt.Key.Key_Home:
            self.reset_view()
            event.accept()
            return
        super().keyPressEvent(event)


class _FinalPreviewDialog(QDialog):
    """解析結果を確認する、Matplotlib のナビゲーション構成を意識したダイアログ。"""

    def __init__(
        self,
        image: np.ndarray,
        mask_regions: list[tuple[int, int, int, int]] | None = None,
        display_options: dict | None = None,
        invalid_sunspot_mask: np.ndarray | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """ツールバー、画像キャンバス、マスク表示、座標 HUD を構築する。

        Parameters
        ----------
        image : numpy.ndarray
            最終確認する2次元画像。
        mask_regions : list[tuple[int, int, int, int]] | None, optional
            左上原点のマスク矩形。
        display_options : dict | None, optional
            最終確認画面の表示設定。
        invalid_sunspot_mask : numpy.ndarray | None, optional
            マスクで除外された黒点画素を True とした2次元配列。
        parent : QWidget | None, optional
            親ウィジェット。
        """
        super().__init__(parent)
        self.setWindowTitle("Final Result Preview")

        self.canvas = _FinalPreviewCanvas(
            image,
            mask_regions,
            display_options,
            invalid_sunspot_mask,
            solar_geometry,
            rmask_radius_ratio,
        )
        toolbar = QToolBar(self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        home_action = QAction("Home", self)
        home_action.setToolTip("画像全体を表示 (Home)")
        home_action.setShortcut(QKeySequence(Qt.Key.Key_Home))
        home_action.triggered.connect(self.canvas.reset_view)
        toolbar.addAction(home_action)

        back_action = QAction("Back", self)
        back_action.setToolTip("1つ前の表示範囲へ戻る")
        back_action.setEnabled(False)
        back_action.triggered.connect(self.canvas.go_back)
        toolbar.addAction(back_action)
        forward_action = QAction("Forward", self)
        forward_action.setToolTip("次の表示範囲へ進む")
        forward_action.setEnabled(False)
        forward_action.triggered.connect(self.canvas.go_forward)
        toolbar.addAction(forward_action)
        toolbar.addSeparator()

        self.canvas.historyChanged.connect(
            lambda can_go_back, can_go_forward: (
                back_action.setEnabled(can_go_back),
                forward_action.setEnabled(can_go_forward),
            )
        )
        self.canvas._emit_history_state()

        navigation_group = QActionGroup(self)
        navigation_group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
        pan_action = QAction("Pan", self)
        pan_action.setCheckable(True)
        pan_action.setToolTip("左ドラッグで移動。Shift + 右ドラッグでも移動できます。")
        zoom_action = QAction("Zoom", self)
        zoom_action.setCheckable(True)
        zoom_action.setToolTip("左ドラッグで範囲を選択して拡大します。")
        navigation_group.addAction(pan_action)
        navigation_group.addAction(zoom_action)
        toolbar.addAction(pan_action)
        toolbar.addAction(zoom_action)

        toolbar.addSeparator()
        mask_fill_checkbox = QCheckBox("Mask Fill", self)
        mask_fill_checkbox.setChecked(
            bool((display_options or {}).get("show_mask_fill", False))
        )
        mask_fill_checkbox.setToolTip("選択したマスク範囲の塗りつぶしを表示／非表示")
        mask_fill_checkbox.toggled.connect(self.canvas.set_mask_fill_visible)
        toolbar.addWidget(mask_fill_checkbox)

        rmask_fill_checkbox = QCheckBox("Rmask Fill", self)
        rmask_fill_checkbox.setChecked(
            bool((display_options or {}).get("show_rmask_fill", False))
        )
        rmask_fill_checkbox.setEnabled(solar_geometry is not None)
        rmask_fill_checkbox.setToolTip("Rmask の除外領域（境界円の外側）の塗りつぶしを表示／非表示")
        rmask_fill_checkbox.toggled.connect(self.canvas.set_rmask_fill_visible)
        toolbar.addWidget(rmask_fill_checkbox)

        solar_circle_checkbox = QCheckBox("太陽近似円", self)
        solar_circle_checkbox.setChecked(
            bool((display_options or {}).get("show_solar_circle", False))
        )
        solar_circle_checkbox.setEnabled(solar_geometry is not None)
        solar_circle_checkbox.setToolTip("MIN2 が推定した太陽近似円と中心位置を表示／非表示")
        solar_circle_checkbox.toggled.connect(self.canvas.set_solar_circle_visible)
        toolbar.addWidget(solar_circle_checkbox)

        invalid_sunspots_checkbox = QCheckBox("無効黒点", self)
        invalid_sunspots_checkbox.setChecked(
            bool((display_options or {}).get("show_invalid_sunspots", True))
        )
        invalid_sunspots_checkbox.setToolTip(
            "マスクで除外された黒点を赤色で表示／非表示"
        )
        invalid_sunspots_checkbox.toggled.connect(
            self.canvas.set_invalid_sunspots_visible
        )
        toolbar.addWidget(invalid_sunspots_checkbox)

        pan_action.toggled.connect(
            lambda checked: self.canvas.set_navigation_mode("pan" if checked else "none")
        )
        zoom_action.toggled.connect(
            lambda checked: self.canvas.set_navigation_mode("zoom" if checked else "none")
        )

        help_label = QLabel(" | Ctrl + wheel: Zoom   |   Shift + right drag: Pan")
        toolbar.addWidget(help_label)

        cursor_label = QLabel("Cursor: x=—, y=—")
        range_label = QLabel()
        cursor_label.setMinimumWidth(180)
        range_label.setMinimumWidth(300)
        self.canvas.cursorChanged.connect(cursor_label.setText)
        self.canvas.viewChanged.connect(range_label.setText)
        self.canvas._emit_view_range()

        status_layout = QHBoxLayout()
        status_layout.addWidget(cursor_label)
        status_layout.addStretch(1)
        status_layout.addWidget(range_label)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(status_layout)
        layout.addWidget(button_box)
        self.setMinimumSize(640, 480)
        self.resize(1100, 820)


def show_final_preview(
    image: np.ndarray,
    mask_regions: list[tuple[int, int, int, int]] | None = None,
    display_options: dict | None = None,
    invalid_sunspot_mask: np.ndarray | None = None,
    solar_geometry: tuple[tuple[float, float], float] | None = None,
    rmask_radius_ratio: float = 0.85,
) -> None:
    """PyQt ダイアログで最終解析画像とマスク領域を操作・確認する。

    Parameters
    ----------
    image : numpy.ndarray
        表示する2次元の最終解析画像。
    mask_regions : list[tuple[int, int, int, int]] | None, optional
        左上原点のマスク矩形。
    display_options : dict | None, optional
        マスク塗りつぶし・無効黒点表示の初期状態、色、不透明度を含む表示設定。
    invalid_sunspot_mask : numpy.ndarray | None, optional
        マスクで除外された黒点画素を True とした2次元配列。
    """
    ensure_qapplication()
    LOGGER.info(
        "Displaying final result preview: shape=%s, dtype=%s",
        image.shape,
        image.dtype,
    )
    dialog = _FinalPreviewDialog(
        image,
        mask_regions,
        display_options,
        invalid_sunspot_mask,
        solar_geometry,
        rmask_radius_ratio,
    )
    dialog.exec()
    LOGGER.info("Final result preview closed.")
