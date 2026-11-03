"""黒点候補を矩形領域によって除外するモジュール。"""

import logging
from typing import Any, Literal, overload

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
    QPolygonF,
    QResizeEvent,
)
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.package.qt_dialogs import ensure_qapplication
from src.package.qt_histogram import BrightnessHistogramWidget
from src.package.qt_image import colorize_intensity

LOGGER = logging.getLogger("sunspot.mask")


def component_polygons(
    label_image: np.ndarray,
    selected_labels: list[int] | set[int],
) -> list[np.ndarray]:
    """Convert selected connected components to pixel-coordinate outline polygons."""
    if label_image.ndim != 2:
        raise ValueError("ラベル画像は2次元配列にしてください。")
    polygons: list[np.ndarray] = []
    for label_value in sorted({int(value) for value in selected_labels if int(value) > 0}):
        component = (label_image == label_value).astype(np.uint8)
        if not np.any(component):
            continue
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = contour.reshape(-1, 2).astype(np.float64)
            if len(points) < 3:
                ys, xs = np.nonzero(component)
                if not xs.size:
                    continue
                points = np.asarray(
                    [
                        [xs.min() - 0.5, ys.min() - 0.5],
                        [xs.max() + 0.5, ys.min() - 0.5],
                        [xs.max() + 0.5, ys.max() + 0.5],
                        [xs.min() - 0.5, ys.max() + 0.5],
                    ],
                    dtype=np.float64,
                )
            polygons.append(points)
    return polygons


def component_selection_details(
    label_image: np.ndarray,
    selected_labels: list[int] | set[int],
) -> list[dict[str, Any]]:
    """Return stable label, geometry, and polygon details for selected components."""
    if label_image.ndim != 2:
        raise ValueError("ラベル画像は2次元配列にしてください。")
    details: list[dict[str, Any]] = []
    for index, label_value in enumerate(
        sorted({int(value) for value in selected_labels if int(value) > 0}),
        start=1,
    ):
        ys, xs = np.nonzero(label_image == label_value)
        if not xs.size:
            continue
        polygons = component_polygons(label_image, {label_value})
        details.append(
            {
                "index": index,
                "selection_method": "connected_component",
                "label": label_value,
                "x": int(xs.min()),
                "y": int(ys.min()),
                "width": int(xs.max() - xs.min() + 1),
                "height": int(ys.max() - ys.min() + 1),
                "area_pixels": int(xs.size),
                "centroid_x_px": float(xs.mean()),
                "centroid_y_px": float(ys.mean()),
                "excluded_candidate_labels": [label_value],
                "polygons": [polygon.tolist() for polygon in polygons],
            }
        )
    return details


def _to_display_image(image: np.ndarray, colormap: str) -> np.ndarray:
    """輝度画像を指定カラーマップの表示用 BGR 画像へ変換する。"""
    display_image, _, _ = colorize_intensity(image, colormap)
    return display_image


class _MaskCanvas(QWidget):
    """画像とマスク矩形を QPainter で描画するキャンバス。"""

    regionsChanged = pyqtSignal(int)

    def __init__(
        self,
        image: np.ndarray,
        display_image: np.ndarray,
        options: dict,
        parent: QWidget | None = None,
        *,
        read_only: bool = False,
        north_vector: tuple[float, float] | None = None,
        flip_x: bool = False,
        label_image: np.ndarray | None = None,
        selection_mode: str = "rectangle",
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
        rgb_bytes = rgb_image.tobytes()
        self._qimage = QImage(
            rgb_bytes,
            self.image_width,
            self.image_height,
            int(rgb_image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()

        finite_values = image[np.isfinite(image)]
        if finite_values.size:
            self._minimum = float(finite_values.min())
            self._maximum = float(finite_values.max())
        else:
            self._minimum = 0.0
            self._maximum = 0.0

        # 画像と同じ OpenCV カラーマップで、上端が最大値・下端が最小値の
        # 縦向きカラーバーを作成する。
        gradient_values = np.arange(255, -1, -1, dtype=np.uint8).reshape(256, 1)
        gradient_bgr = _to_display_image(
            gradient_values,
            str(options.get("colormap", "HOT")),
        )
        gradient_rgb = cv2.cvtColor(gradient_bgr, cv2.COLOR_BGR2RGB)
        self._colorbar_qimage = QImage(
            gradient_rgb.tobytes(),
            1,
            256,
            int(gradient_rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()

        self._options = options
        self._read_only = read_only
        self._north_vector = north_vector
        self._flip_x = bool(flip_x)
        self._preview_polygons: list[np.ndarray] = []
        self._preview_hover_polygons: list[np.ndarray] = []
        self._label_image = None
        if label_image is not None:
            if label_image.ndim != 2 or label_image.shape != image.shape:
                raise ValueError("連結成分ラベル画像は表示画像と同じ形状の2次元配列にしてください。")
            self._label_image = np.asarray(label_image)
        self._selection_mode = selection_mode
        if self._selection_mode not in ("rectangle", "connected_component"):
            raise ValueError(f"未対応のマスク選択方式です: {self._selection_mode}")
        if self._selection_mode == "connected_component" and self._label_image is None:
            raise ValueError("連結成分選択にはラベル画像が必要です。")
        self.selected_labels: set[int] = set()
        self.hovered_label: int | None = None
        self._selection_history: list[set[int]] = []
        self._selected_overlay: QImage | None = None
        self._hover_overlay: QImage | None = None
        self.regions: list[tuple[int, int, int, int]] = []
        self._drawing = False
        self._start_point = (0, 0)
        self._current_point = (0, 0)
        self._mouse_point: tuple[int, int] | None = None
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(
            Qt.CursorShape.ArrowCursor if read_only else Qt.CursorShape.CrossCursor
        )

    def set_preview_polygons(
        self,
        polygons: list[np.ndarray],
        hover_polygons: list[np.ndarray] | None = None,
    ) -> None:
        """Replace synchronized selected and hovered polygon overlays."""
        self._preview_polygons = [
            np.asarray(polygon, dtype=np.float64)
            for polygon in polygons
            if np.asarray(polygon).ndim == 2 and np.asarray(polygon).shape[1] == 2
        ]
        self._preview_hover_polygons = [
            np.asarray(polygon, dtype=np.float64)
            for polygon in (hover_polygons or [])
            if np.asarray(polygon).ndim == 2 and np.asarray(polygon).shape[1] == 2
        ]
        self.update()

    def _make_label_overlay(
        self,
        label_values: set[int],
        color: tuple[int, int, int, int],
    ) -> QImage | None:
        """Create a cached transparent overlay for one or more labels."""
        if self._label_image is None or not label_values:
            return None
        overlay = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)
        overlay[np.isin(self._label_image, tuple(label_values))] = color
        return QImage(
            overlay.tobytes(),
            self.image_width,
            self.image_height,
            int(overlay.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()

    def _refresh_component_overlays(self) -> None:
        """Refresh cached selected and hover overlays after an interaction."""
        self._selected_overlay = self._make_label_overlay(
            self.selected_labels, (255, 64, 64, 145)
        )
        hover_values = (
            {self.hovered_label}
            if self.hovered_label is not None
            and self.hovered_label > 0
            and self.hovered_label not in self.selected_labels
            else set()
        )
        self._hover_overlay = self._make_label_overlay(
            hover_values, (255, 220, 0, 170)
        )

    def selected_component_polygons(self) -> list[np.ndarray]:
        """Return polygon outlines for all selected components."""
        if self._label_image is None:
            return []
        return component_polygons(self._label_image, self.selected_labels)

    def hovered_component_polygons(self) -> list[np.ndarray]:
        """Return polygon outlines for the current unselected hover component."""
        if self._label_image is None or self.hovered_label is None:
            return []
        if self.hovered_label in self.selected_labels or self.hovered_label <= 0:
            return []
        return component_polygons(self._label_image, {self.hovered_label})

    def preview_regions(self) -> list[tuple[int, int, int, int]]:
        """Return committed regions plus the rectangle currently being dragged."""
        regions = self.regions.copy()
        if self._drawing:
            x1, y1 = self._start_point
            x2, y2 = self._current_point
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            if width > 0 and height > 0:
                regions.append((min(x1, x2), min(y1, y2), width, height))
        return regions

    def _draw_orientation_axes(self, painter: QPainter, target: QRectF) -> None:
        """Draw Cartesian +X and solar-north +Y arrows over the image."""
        if self._north_vector is None:
            return
        north_cartesian = np.asarray(self._north_vector, dtype=np.float64)
        norm = float(np.linalg.norm(north_cartesian))
        if not np.isfinite(norm) or norm == 0:
            return
        north_cartesian /= norm
        north_image = np.asarray(
            [north_cartesian[0], -north_cartesian[1]],
            dtype=np.float64,
        )
        positive_x = np.asarray([-north_image[1], north_image[0]], dtype=np.float64)
        if self._flip_x:
            positive_x *= -1.0
        origin = QPointF(target.left() + 58.0, target.top() + 62.0)
        axis_length = 42.0
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for vector, color, label in (
            (positive_x, QColor(80, 220, 255), "+X"),
            (north_image, QColor(255, 220, 70), "+Y 北"),
        ):
            endpoint = QPointF(
                origin.x() + float(vector[0]) * axis_length,
                origin.y() + float(vector[1]) * axis_length,
            )
            pen = QPen(color, 3)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(origin, endpoint)
            painter.drawEllipse(endpoint, 2.5, 2.5)
            painter.drawText(endpoint + QPointF(5.0, -5.0), label)
        painter.restore()

    def _image_rect(self) -> QRectF:
        """キャンバス内で画像をアスペクト比を保って配置する領域を返す。

        Returns
        -------
        QRectF
            画像の描画先領域。
        """
        # 右側にカラーバーと数値ラベルを確保し、画像の四辺にも余白を設ける。
        margin = 20
        plot_width = max(1, self.width() - 116 - margin * 2)
        plot_height = max(1, self.height() - margin * 2)
        scale = min(
            plot_width / self.image_width,
            plot_height / self.image_height,
        )
        shown_width = self.image_width * scale
        shown_height = self.image_height * scale
        return QRectF(
            margin + (plot_width - shown_width) / 2,
            margin + (plot_height - shown_height) / 2,
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
            if self._selection_mode == "connected_component":
                if self.hovered_label is not None and self.hovered_label > 0:
                    hover_area = int(np.count_nonzero(self._label_image == self.hovered_label))
                    lines.append(
                        f"Hovered component: label={self.hovered_label}, area={hover_area} px"
                    )
                lines.append(f"Selected components: {len(self.selected_labels)}")
            else:
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
            if self._selection_mode == "connected_component":
                lines.append("Hover: highlight component | Click: toggle exclusion | Backspace: undo")
            else:
                lines.append("Drag: select ROI | Enter/Esc: finish | Backspace: undo")
        if self._options["show_scale"]:
            lines.append(f"Display scale: {scale * 100:.1f}%")
        return lines

    def paintEvent(self, event: QPaintEvent | None) -> None:
        """画像、矩形、座標情報を Qt のベクター描画で再描画する。

        Parameters
        ----------
        event : QPaintEvent | None
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
        if self._selected_overlay is not None:
            painter.drawImage(target, self._selected_overlay)
        if self._hover_overlay is not None:
            painter.drawImage(target, self._hover_overlay)

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
        self._draw_orientation_axes(painter, target)

        colorbar = QRectF(target.right() + 14, target.top(), 16, target.height())
        painter.drawImage(colorbar, self._colorbar_qimage)
        colorbar_pen = QPen(QColor(235, 235, 235), 1)
        colorbar_pen.setCosmetic(True)
        painter.setPen(colorbar_pen)
        painter.drawRect(colorbar)
        painter.setPen(QColor(240, 240, 240))
        painter.drawText(
            QRectF(colorbar.right() + 5, colorbar.top() - 8, 78, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self._maximum:g}",
        )
        painter.drawText(
            QRectF(colorbar.right() + 5, colorbar.bottom() - 12, 78, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self._minimum:g}",
        )

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

        if self._preview_polygons or self._preview_hover_polygons:
            for polygon_group, pen_color, fill_color in (
                (self._preview_polygons, QColor(255, 64, 64, 230), QColor(255, 64, 64, 100)),
                (self._preview_hover_polygons, QColor(255, 220, 0, 240), QColor(255, 220, 0, 85)),
            ):
                preview_pen = QPen(pen_color, 2)
                preview_pen.setCosmetic(True)
                painter.setPen(preview_pen)
                painter.setBrush(fill_color)
                for polygon_points in polygon_group:
                    polygon = QPolygonF(
                        [
                            QPointF(
                                target.left() + float(point[0]) * scale_x,
                                target.top() + float(point[1]) * scale_y,
                            )
                            for point in polygon_points
                        ]
                    )
                    if polygon.size() >= 3:
                        painter.drawPolygon(polygon)
            painter.setBrush(Qt.BrushStyle.NoBrush)

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

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """左クリック位置から矩形選択を開始する。

        Parameters
        ----------
        event : QMouseEvent | None
            マウス押下イベント。
        """
        if (
            self._read_only
            or event is None
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return
        point = self._map_to_image(event.position())
        if point is None:
            return
        self.setFocus()
        if self._selection_mode == "connected_component":
            assert self._label_image is not None
            label_value = int(self._label_image[point[1], point[0]])
            if label_value > 0:
                self._selection_history.append(self.selected_labels.copy())
                if label_value in self.selected_labels:
                    self.selected_labels.remove(label_value)
                else:
                    self.selected_labels.add(label_value)
                self._refresh_component_overlays()
                self.regionsChanged.emit(len(self.selected_labels))
            self.update()
            return
        self._drawing = True
        self._start_point = point
        self._current_point = point
        self._mouse_point = point
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """カーソル座標とドラッグ中の矩形終点を更新する。

        Parameters
        ----------
        event : QMouseEvent | None
            マウス移動イベント。
        """
        if event is None:
            return
        if self._read_only:
            self._mouse_point = self._map_to_image(event.position())
            self.update()
            return
        if self._selection_mode == "connected_component":
            point = self._map_to_image(event.position())
            self._mouse_point = point
            new_label = (
                int(self._label_image[point[1], point[0]])
                if point is not None and self._label_image is not None
                else None
            )
            if new_label != self.hovered_label:
                self.hovered_label = new_label
                self._refresh_component_overlays()
                self.regionsChanged.emit(len(self.selected_labels))
            self.update()
            return
        point = self._map_to_image(event.position(), clamp=self._drawing)
        self._mouse_point = point
        if self._drawing and point is not None:
            self._current_point = point
            self.regionsChanged.emit(len(self.regions))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        """ドラッグを終了し、有効な矩形を選択領域へ追加する。

        Parameters
        ----------
        event : QMouseEvent | None
            マウス解放イベント。
        """
        if (
            self._read_only
            or event is None
            or event.button() != Qt.MouseButton.LeftButton
            or self._selection_mode == "connected_component"
            or not self._drawing
        ):
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
        self._drawing = False
        if point is not None:
            self.regionsChanged.emit(len(self.regions))
        self.update()

    def undo(self) -> None:
        """最後に追加した矩形を取り消す。

        Returns
        -------
        None
        """
        if self._selection_mode == "connected_component":
            if self._selection_history:
                self.selected_labels = self._selection_history.pop()
                self._refresh_component_overlays()
                self.regionsChanged.emit(len(self.selected_labels))
        elif self.regions:
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
        *,
        preview_images: list[tuple[str, np.ndarray, Any]] | None = None,
        editor_transform: Any | None = None,
        label_image: np.ndarray | None = None,
        selection_mode: str = "rectangle",
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
        self.setWindowTitle(
            "Select Connected Components"
            if selection_mode == "connected_component"
            else "Select Mask Regions"
        )
        self._editor_transform = editor_transform
        self.preview_canvases: list[tuple[_MaskCanvas, Any]] = []

        editor_north = None
        editor_flip_x = False
        if editor_transform is not None:
            editor_north = tuple(editor_transform.orientation.north_vector)
            editor_flip_x = bool(editor_transform.orientation.flip_x)
        self.canvas = _MaskCanvas(
            image,
            _to_display_image(image, str(options["colormap"])),
            options,
            north_vector=editor_north,
            flip_x=editor_flip_x,
            label_image=label_image,
            selection_mode=selection_mode,
        )
        self.region_count_label = QLabel(
            "選択成分: 0" if selection_mode == "connected_component" else "選択領域: 0"
        )
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
            lambda count: self.region_count_label.setText(
                f"選択成分: {count}"
                if selection_mode == "connected_component"
                else f"選択領域: {count}"
            )
        )

        footer = QHBoxLayout()
        footer.addWidget(self.region_count_label)
        footer.addStretch(1)
        footer.addWidget(undo_button)
        footer.addWidget(button_box)

        self.histogram = BrightnessHistogramWidget(image, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        if preview_images:
            preview_grid = QGridLayout()
            preview_grid.setContentsMargins(0, 0, 0, 0)
            preview_grid.setSpacing(8)
            for index in range(4):
                row, column = divmod(index, 2)
                if index == 0:
                    title = preview_images[0][0]
                    canvas = self.canvas
                elif index < len(preview_images):
                    title, preview_image, preview_transform = preview_images[index]
                    preview_north = tuple(preview_transform.orientation.north_vector)
                    canvas = _MaskCanvas(
                        preview_image,
                        _to_display_image(
                            preview_image,
                            str(options["colormap"]),
                        ),
                        options,
                        read_only=True,
                        north_vector=preview_north,
                        flip_x=bool(preview_transform.orientation.flip_x),
                    )
                    self.preview_canvases.append((canvas, preview_transform))
                else:
                    placeholder = QLabel("画像未提供")
                    placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    placeholder.setMinimumSize(320, 240)
                    placeholder.setStyleSheet(
                        "background: #303030; color: #cccccc; border: 1px solid #666666;"
                    )
                    preview_grid.addWidget(placeholder, row, column)
                    continue

                pane = QWidget(self)
                pane_layout = QVBoxLayout(pane)
                pane_layout.setContentsMargins(2, 2, 2, 2)
                title_label = QLabel(
                    f"{title} — {'編集' if index == 0 else '同期プレビュー'}"
                )
                title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pane_layout.addWidget(title_label)
                pane_layout.addWidget(canvas, 1)
                preview_grid.addWidget(pane, row, column)
            self.canvas.regionsChanged.connect(self._sync_preview_masks)
            layout.addLayout(preview_grid, 1)
        else:
            layout.addWidget(self.canvas, 1)
        layout.addWidget(self.histogram)
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

    def _sync_preview_masks(self, _region_count: int) -> None:
        """Synchronize rectangle or component overlays across comparison images."""
        if self._editor_transform is None:
            return
        if self.canvas._selection_mode == "connected_component":
            selected_polygons = [
                self._editor_transform.pixel_points_to_normalized(polygon)
                for polygon in self.canvas.selected_component_polygons()
            ]
            hover_polygons = [
                self._editor_transform.pixel_points_to_normalized(polygon)
                for polygon in self.canvas.hovered_component_polygons()
            ]
        else:
            selected_polygons = [
                self._editor_transform.rectangle_to_normalized_polygon(region)
                for region in self.canvas.preview_regions()
            ]
            hover_polygons = []
        for preview_canvas, preview_transform in self.preview_canvases:
            pixel_polygons = [
                preview_transform.normalized_points_to_pixel(polygon)
                for polygon in selected_polygons
            ]
            hover_pixel_polygons = [
                preview_transform.normalized_points_to_pixel(polygon)
                for polygon in hover_polygons
            ]
            preview_canvas.set_preview_polygons(pixel_polygons, hover_pixel_polygons)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        """ユーザーによるリサイズ時にウィンドウの縦横比を維持する。

        Parameters
        ----------
        event : QResizeEvent | None
            Qt のリサイズイベント。
        """
        if event is None:
            return
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

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """ウィンドウを閉じた場合も現在の矩形選択を確定する。

        Parameters
        ----------
        event : QCloseEvent | None
            Qt のウィンドウ終了イベント。

        Returns
        -------
        None
        """
        self.done(QDialog.DialogCode.Accepted)
        if event is not None:
            event.accept()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        """Enter、Backspace、Esc キーの操作を処理する。

        Parameters
        ----------
        event : QKeyEvent | None
            キー押下イベント。
        """
        if event is None:
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
        elif event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.canvas.undo()
            self.region_count_label.setText(
                f"選択成分: {len(self.canvas.selected_labels)}"
                if self.canvas._selection_mode == "connected_component"
                else f"選択領域: {len(self.canvas.regions)}"
            )
        elif event.key() == Qt.Key.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)


def selectMaskRegions(
    image: np.ndarray,
    display_options: dict | None = None,
    *,
    preview_images: list[tuple[str, np.ndarray, Any]] | None = None,
    editor_transform: Any | None = None,
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
    dialog = _MaskDialog(
        image,
        options,
        preview_images=preview_images,
        editor_transform=editor_transform,
    )
    regions = (
        dialog.canvas.regions.copy()
        if dialog.exec() == QDialog.DialogCode.Accepted
        else []
    )

    LOGGER.info("合計 %d 個のマスク領域を選択しました。", len(regions))
    for index, (x, y, width, height) in enumerate(regions, start=1):
        LOGGER.info(
            "マスク領域 %d: x=%d, y=%d, width=%d, height=%d",
            index,
            x,
            y,
            width,
            height,
        )
    return regions



def selectMaskComponents(
    image: np.ndarray,
    label_image: np.ndarray,
    display_options: dict | None = None,
    *,
    preview_images: list[tuple[str, np.ndarray, Any]] | None = None,
    editor_transform: Any | None = None,
) -> dict[str, Any]:
    """Open the connected-component picker and return selected labels and polygons."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("入力画像は空でない2次元グレースケール画像にしてください。")
    if label_image.ndim != 2 or label_image.shape != image.shape:
        raise ValueError("連結成分ラベル画像は入力画像と同じ形状の2次元配列にしてください。")
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
    dialog = _MaskDialog(
        image,
        options,
        preview_images=preview_images,
        editor_transform=editor_transform,
        label_image=label_image,
        selection_mode="connected_component",
    )
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    selected_labels = sorted(dialog.canvas.selected_labels) if accepted else []
    components = component_selection_details(label_image, selected_labels)
    selected_polygons = [
        polygon for component in components for polygon in component["polygons"]
    ]
    LOGGER.info("手動選択した連結成分数: %d", len(selected_labels))
    for component in components:
        LOGGER.info(
            "連結成分を手動除外: label=%d, area=%d, centroid=(%.2f, %.2f)",
            component["label"],
            component["area_pixels"],
            component["centroid_x_px"],
            component["centroid_y_px"],
        )
    return {
        "selected_labels": selected_labels,
        "components": components,
        "polygons": selected_polygons,
    }


@overload
def applyComponentMask(
    labels: np.ndarray,
    selected_labels: list[int] | set[int],
    original_count: int,
    return_details: Literal[False] = False,
) -> tuple[int, np.ndarray]: ...


@overload
def applyComponentMask(
    labels: np.ndarray,
    selected_labels: list[int] | set[int],
    original_count: int,
    return_details: Literal[True],
) -> tuple[int, np.ndarray, dict]: ...


def applyComponentMask(
    labels: np.ndarray,
    selected_labels: list[int] | set[int],
    original_count: int,
    return_details: bool = False,
) -> tuple[int, np.ndarray] | tuple[int, np.ndarray, dict]:
    """Exclude exactly the selected connected-component labels."""
    if labels.ndim != 2:
        raise ValueError("ラベル画像は2次元配列にしてください。")
    available = {int(value) for value in np.unique(labels) if int(value) > 0}
    excluded_labels = sorted(
        {int(value) for value in selected_labels if int(value) > 0}.intersection(available)
    )
    regions = component_selection_details(labels, excluded_labels)
    masked_labels = labels.copy()
    if excluded_labels:
        masked_labels[np.isin(masked_labels, excluded_labels)] = 0
    remaining_count = max(0, int(original_count) - len(excluded_labels))
    masked_image = np.where(masked_labels > 0, 0, 255).astype(np.uint8)
    LOGGER.info("連結成分クリック選択で除外した黒点数: %d", len(excluded_labels))
    LOGGER.info("連結成分マスク適用後の黒点数: %d", remaining_count)
    if not return_details:
        return remaining_count, masked_image

    height, width = labels.shape
    for region in regions:
        x1 = max(0, int(region["x"]))
        y1 = max(0, int(region["y"]))
        x2 = min(width, int(region["x"] + region["width"]))
        y2 = min(height, int(region["y"] + region["height"]))
        region["clipped_bounds"] = {
            "x1": x1,
            "y1": y1,
            "x2_exclusive": x2,
            "y2_exclusive": y2,
        }
        region["image_coverage_percent"] = round(
            region["area_pixels"] / (width * height) * 100, 6
        )
        region["overlapping_candidate_count"] = 1
        region["overlapping_candidate_labels"] = [int(region["label"])]
        region["excluded_candidate_count"] = 1
    details = {
        "image_width": int(width),
        "image_height": int(height),
        "selected_region_count": len(regions),
        "selection_method": "connected_component",
        "boundary_policy": "exact_component",
        "regions": regions,
        "original_candidate_count": int(original_count),
        "overlapping_candidate_labels": excluded_labels,
        "excluded_candidate_count": len(excluded_labels),
        "excluded_candidate_labels": excluded_labels,
        "remaining_candidate_count": int(remaining_count),
    }
    return remaining_count, masked_image, details

@overload
def applyMask(
    labels: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    original_count: int,
    return_details: Literal[False] = False,
    boundary_policy: Literal["overlap", "centroid", "contained"] = "overlap",
) -> tuple[int, np.ndarray]: ...


@overload
def applyMask(
    labels: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    original_count: int,
    return_details: Literal[True],
    boundary_policy: Literal["overlap", "centroid", "contained"] = "overlap",
) -> tuple[int, np.ndarray, dict]: ...


def applyMask(
    labels: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    original_count: int,
    return_details: bool = False,
    boundary_policy: Literal["overlap", "centroid", "contained"] = "overlap",
) -> tuple[int, np.ndarray] | tuple[int, np.ndarray, dict]:
    """矩形マスクと重なる黒点候補を、指定した境界判定で除外する。

    overlap は矩形内に1画素でも入った候補を除外する従来方式。
    centroid は候補の重心が矩形内にある場合だけ除外する方式。
    contained はすべての矩形領域の和集合に全画素が含まれる候補を除外する方式。
    """
    if labels.ndim != 2:
        raise ValueError("ラベル画像は2次元配列にしてください。")
    if boundary_policy not in ("overlap", "centroid", "contained"):
        raise ValueError(f"未対応のマスク境界判定方式です: {boundary_policy}")

    height, width = labels.shape
    candidate_labels = np.unique(labels)
    candidate_labels = candidate_labels[candidate_labels > 0]
    candidate_centroids: dict[int, tuple[float, float]] = {}
    for label_value in candidate_labels:
        y_coordinates, x_coordinates = np.nonzero(labels == label_value)
        if x_coordinates.size:
            candidate_centroids[int(label_value)] = (
                float(x_coordinates.mean()),
                float(y_coordinates.mean()),
            )

    # contained は全ROIの和集合を除外領域として扱う。
    # そのため、黒点が複数ROIにまたがっていても、全画素が和集合内なら除外する。
    mask_coverage = np.zeros(labels.shape, dtype=bool)
    for x, y, roi_width, roi_height in regions:
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(width, int(x + roi_width))
        y2 = min(height, int(y + roi_height))
        if x1 < x2 and y1 < y2:
            mask_coverage[y1:y2, x1:x2] = True

    contained_labels: set[int] = set()
    if boundary_policy == "contained":
        for label_value in candidate_labels:
            y_coordinates, x_coordinates = np.nonzero(labels == label_value)
            if x_coordinates.size and np.all(mask_coverage[y_coordinates, x_coordinates]):
                contained_labels.add(int(label_value))

    excluded_labels: set[int] = set()
    physically_overlapping_labels: set[int] = set()
    region_details: list[dict] = []

    for index, (x, y, roi_width, roi_height) in enumerate(regions, start=1):
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(width, int(x + roi_width))
        y2 = min(height, int(y + roi_height))

        if x1 >= x2 or y1 >= y2:
            region_details.append(
                {
                    "index": index,
                    "x": int(x),
                    "y": int(y),
                    "width": int(roi_width),
                    "height": int(roi_height),
                    "clipped_bounds": None,
                    "area_pixels": 0,
                    "image_coverage_percent": 0.0,
                    "overlapping_candidate_count": 0,
                    "overlapping_candidate_labels": [],
                    "excluded_candidate_count": 0,
                    "excluded_candidate_labels": [],
                }
            )
            continue

        overlapping_values = np.unique(labels[y1:y2, x1:x2])
        overlapping_labels = {
            int(label_value) for label_value in overlapping_values if label_value > 0
        }
        physically_overlapping_labels.update(overlapping_labels)

        if boundary_policy == "overlap":
            selected_labels = overlapping_labels
        elif boundary_policy == "centroid":
            selected_labels = {
                label_value
                for label_value, (centroid_x, centroid_y) in candidate_centroids.items()
                if x1 <= centroid_x < x2 and y1 <= centroid_y < y2
            }
        else:
            # 複数ROIの和集合で全画素が覆われる黒点を、各重複ROIの詳細にも記録する。
            selected_labels = contained_labels.intersection(overlapping_labels)

        excluded_labels.update(selected_labels)
        area_pixels = (x2 - x1) * (y2 - y1)
        region_details.append(
            {
                "index": index,
                "x": int(x),
                "y": int(y),
                "width": int(roi_width),
                "height": int(roi_height),
                "clipped_bounds": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2_exclusive": int(x2),
                    "y2_exclusive": int(y2),
                },
                "area_pixels": int(area_pixels),
                "image_coverage_percent": round(
                    area_pixels / (width * height) * 100, 4
                ),
                "overlapping_candidate_count": len(overlapping_labels),
                "overlapping_candidate_labels": sorted(overlapping_labels),
                "excluded_candidate_count": len(selected_labels),
                "excluded_candidate_labels": sorted(selected_labels),
            }
        )

    masked_labels = labels.copy()
    if excluded_labels:
        masked_labels[np.isin(masked_labels, list(excluded_labels))] = 0

    count = max(0, int(original_count) - len(excluded_labels))
    LOGGER.info(
        "矩形マスクで除外した黒点数: %d (boundary_policy=%s)",
        len(excluded_labels),
        boundary_policy,
    )
    LOGGER.info("矩形マスク適用後の黒点数: %d", count)

    masked_image = np.where(masked_labels > 0, 0, 255).astype(np.uint8)
    if not return_details:
        return count, masked_image

    details = {
        "image_width": int(width),
        "image_height": int(height),
        "selected_region_count": len(regions),
        "boundary_policy": boundary_policy,
        "regions": region_details,
        "original_candidate_count": int(original_count),
        "overlapping_candidate_labels": sorted(physically_overlapping_labels),
        "excluded_candidate_count": len(excluded_labels),
        "excluded_candidate_labels": sorted(excluded_labels),
        "remaining_candidate_count": int(count),
    }
    return count, masked_image, details


def _single_polygon_coverage(
    shape: tuple[int, int],
    vertices: np.ndarray,
) -> np.ndarray:
    """Test image pixel centers against one polygon using an even-odd rule."""
    height, width = int(shape[0]), int(shape[1])
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError("多角形マスクは3点以上の (x, y) 座標で指定してください。")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("多角形マスクの頂点には有限値を指定してください。")

    vertices = np.clip(
        vertices,
        [-width * 4, -height * 4],
        [width * 5, height * 5],
    )
    x_min = max(0, int(np.floor(vertices[:, 0].min())))
    x_max = min(width, int(np.ceil(vertices[:, 0].max())) + 1)
    y_min = max(0, int(np.floor(vertices[:, 1].min())))
    y_max = min(height, int(np.ceil(vertices[:, 1].max())) + 1)
    coverage = np.zeros((height, width), dtype=bool)
    if x_min >= x_max or y_min >= y_max:
        return coverage

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, dtype=np.float64),
        np.arange(y_min, y_max, dtype=np.float64),
    )
    inside = np.zeros(grid_x.shape, dtype=bool)
    on_boundary = np.zeros(grid_x.shape, dtype=bool)
    for index in range(len(vertices)):
        first = vertices[index]
        second = vertices[(index + 1) % len(vertices)]
        x1, y1 = float(first[0]), float(first[1])
        x2, y2 = float(second[0]), float(second[1])
        delta_x, delta_y = x2 - x1, y2 - y1
        relative_x, relative_y = grid_x - x1, grid_y - y1
        cross = relative_x * delta_y - relative_y * delta_x
        dot = relative_x * delta_x + relative_y * delta_y
        length_squared = delta_x * delta_x + delta_y * delta_y
        on_boundary |= (
            (np.abs(cross) <= 1e-9)
            & (dot >= -1e-9)
            & (dot <= length_squared + 1e-9)
        )

        if delta_y != 0.0:
            crosses_ray = (y1 > grid_y) != (y2 > grid_y)
            intersection_x = x1 + (grid_y - y1) * delta_x / delta_y
            inside ^= crosses_ray & (grid_x < intersection_x)

    coverage[y_min:y_max, x_min:x_max] = inside | on_boundary
    return coverage


def polygon_coverage_mask(
    shape: tuple[int, int],
    polygons: list[np.ndarray],
) -> np.ndarray:
    """Rasterize polygons by testing pixel centers in the image coordinate plane.

    Parameters
    ----------
    shape : tuple[int, int]
        Image height and width.
    polygons : list[numpy.ndarray]
        Polygon vertices in image pixel coordinates.

    Returns
    -------
    numpy.ndarray
        Boolean array identifying pixel centers covered by any polygon.
    """
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("マスク対象の画像サイズは正の値にしてください。")

    coverage = np.zeros((height, width), dtype=bool)
    for polygon in polygons:
        coverage |= _single_polygon_coverage((height, width), polygon)
    return coverage


@overload
def applyPolygonMask(
    labels: np.ndarray,
    polygons: list[np.ndarray],
    original_count: int,
    return_details: Literal[False] = False,
    boundary_policy: Literal["overlap", "centroid", "contained"] = "overlap",
) -> tuple[int, np.ndarray]: ...


@overload
def applyPolygonMask(
    labels: np.ndarray,
    polygons: list[np.ndarray],
    original_count: int,
    return_details: Literal[True],
    boundary_policy: Literal["overlap", "centroid", "contained"] = "overlap",
) -> tuple[int, np.ndarray, dict]: ...


def applyPolygonMask(
    labels: np.ndarray,
    polygons: list[np.ndarray],
    original_count: int,
    return_details: bool = False,
    boundary_policy: Literal["overlap", "centroid", "contained"] = "overlap",
) -> tuple[int, np.ndarray] | tuple[int, np.ndarray, dict]:
    """Exclude sunspot candidates using polygons in image pixel coordinates.

    Parameters
    ----------
    labels : numpy.ndarray
        Two-dimensional labeled candidate image.
    polygons : list[numpy.ndarray]
        Polygon vertices in zero-based image pixel coordinates.
    original_count : int
        Candidate count before applying the masks.
    return_details : bool, optional
        Whether to return per-polygon exclusion details.
    boundary_policy : Literal["overlap", "centroid", "contained"], optional
        Candidate boundary policy, matching the rectangular mask behavior.

    Returns
    -------
    tuple
        Remaining count and binary image, with details when requested.
    """
    if labels.ndim != 2:
        raise ValueError("ラベル画像は2次元配列にしてください。")
    if boundary_policy not in ("overlap", "centroid", "contained"):
        raise ValueError(f"未対応のマスク境界判定方式です: {boundary_policy}")

    height, width = labels.shape
    candidate_labels = np.unique(labels)
    candidate_labels = candidate_labels[candidate_labels > 0]
    candidate_centroids: dict[int, tuple[float, float]] = {}
    for label_value in candidate_labels:
        y_coordinates, x_coordinates = np.nonzero(labels == label_value)
        if x_coordinates.size:
            candidate_centroids[int(label_value)] = (
                float(x_coordinates.mean()),
                float(y_coordinates.mean()),
            )

    polygon_masks: list[np.ndarray] = []
    for polygon in polygons:
        vertices = np.asarray(polygon, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
            raise ValueError("多角形マスクは3点以上の (x, y) 座標で指定してください。")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("多角形マスクの頂点には有限値を指定してください。")
        polygon_masks.append(_single_polygon_coverage((height, width), vertices))

    mask_coverage = (
        np.logical_or.reduce(polygon_masks)
        if polygon_masks
        else np.zeros((height, width), dtype=bool)
    )
    contained_labels: set[int] = set()
    if boundary_policy == "contained":
        for label_value in candidate_labels:
            y_coordinates, x_coordinates = np.nonzero(labels == label_value)
            if x_coordinates.size and np.all(mask_coverage[y_coordinates, x_coordinates]):
                contained_labels.add(int(label_value))

    excluded_labels: set[int] = set()
    physically_overlapping_labels: set[int] = set()
    region_details: list[dict] = []

    for index, (polygon, region_mask) in enumerate(
        zip(polygons, polygon_masks, strict=True),
        start=1,
    ):
        vertices = np.asarray(polygon, dtype=np.float64)
        ys, xs = np.nonzero(region_mask)
        if xs.size:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            overlapping_values = np.unique(labels[region_mask])
            overlapping_labels = {
                int(value) for value in overlapping_values if value > 0
            }
        else:
            x1 = y1 = x2 = y2 = 0
            overlapping_labels = set()

        physically_overlapping_labels.update(overlapping_labels)
        if boundary_policy == "overlap":
            selected_labels = overlapping_labels
        elif boundary_policy == "centroid":
            points = vertices.astype(np.float32)
            selected_labels = {
                label_value
                for label_value, centroid in candidate_centroids.items()
                if cv2.pointPolygonTest(points, centroid, False) >= 0
            }
        else:
            selected_labels = contained_labels.intersection(overlapping_labels)

        excluded_labels.update(selected_labels)
        area_pixels = int(np.count_nonzero(region_mask))
        region_details.append(
            {
                "index": index,
                "x": int(x1),
                "y": int(y1),
                "width": int(max(0, x2 - x1)),
                "height": int(max(0, y2 - y1)),
                "vertices": vertices.tolist(),
                "clipped_bounds": (
                    {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2_exclusive": int(x2),
                        "y2_exclusive": int(y2),
                    }
                    if area_pixels
                    else None
                ),
                "area_pixels": area_pixels,
                "image_coverage_percent": round(area_pixels / (width * height) * 100, 4),
                "overlapping_candidate_count": len(overlapping_labels),
                "overlapping_candidate_labels": sorted(overlapping_labels),
                "excluded_candidate_count": len(selected_labels),
                "excluded_candidate_labels": sorted(selected_labels),
            }
        )

    masked_labels = labels.copy()
    if excluded_labels:
        masked_labels[np.isin(masked_labels, list(excluded_labels))] = 0
    count = max(0, int(original_count) - len(excluded_labels))
    masked_image = np.where(masked_labels > 0, 0, 255).astype(np.uint8)
    LOGGER.info(
        "多角形マスクで除外した黒点数: %d (boundary_policy=%s)",
        len(excluded_labels),
        boundary_policy,
    )

    if not return_details:
        return count, masked_image

    details = {
        "image_width": int(width),
        "image_height": int(height),
        "selected_region_count": len(polygons),
        "boundary_policy": boundary_policy,
        "regions": region_details,
        "original_candidate_count": int(original_count),
        "overlapping_candidate_labels": sorted(physically_overlapping_labels),
        "excluded_candidate_count": len(excluded_labels),
        "excluded_candidate_labels": sorted(excluded_labels),
        "remaining_candidate_count": int(count),
    }
    return count, masked_image, details



@overload
def applyRadialMask(
    labels: np.ndarray,
    original_count: int,
    center: tuple[float, float],
    solar_radius: float,
    radius_ratio: float,
    return_details: Literal[False] = False,
    boundary_policy: Literal["overlap", "centroid", "contained"] = "overlap",
) -> tuple[int, np.ndarray]: ...


@overload
def applyRadialMask(
    labels: np.ndarray,
    original_count: int,
    center: tuple[float, float],
    solar_radius: float,
    radius_ratio: float,
    return_details: Literal[True],
    boundary_policy: Literal["overlap", "centroid", "contained"] = "overlap",
) -> tuple[int, np.ndarray, dict]: ...


def applyRadialMask(
    labels: np.ndarray,
    original_count: int,
    center: tuple[float, float],
    solar_radius: float,
    radius_ratio: float,
    return_details: bool = False,
    boundary_policy: Literal["overlap", "centroid", "contained"] = "overlap",
) -> tuple[int, np.ndarray] | tuple[int, np.ndarray, dict]:
    """MIN2 の太陽中心・半径に基づき、半径比より外側の黒点候補を除外する。

    overlap は候補の1画素でも円境界の外側にあれば除外する。
    centroid は候補の重心が円境界の外側にある場合だけ除外する。
    """
    if labels.ndim != 2:
        raise ValueError("ラベル画像は2次元配列にしてください。")
    if boundary_policy not in ("overlap", "centroid", "contained"):
        raise ValueError(f"未対応のマスク境界判定方式です: {boundary_policy}")
    if not np.isfinite(radius_ratio) or radius_ratio <= 0.0:
        raise ValueError("Rmask.radius_ratio は正の有限値にしてください。")
    if not np.isfinite(solar_radius) or solar_radius <= 0:
        raise ValueError("太陽半径は正の有限値である必要があります。")
    center_x, center_y = float(center[0]), float(center[1])
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        raise ValueError("太陽中心は有限値である必要があります。")

    threshold_radius = float(solar_radius) * float(radius_ratio)
    if not np.isfinite(threshold_radius):
        raise ValueError("Rmask の除外境界半径が有限値の範囲を超えています。")
    candidate_labels = np.unique(labels)
    candidate_labels = candidate_labels[candidate_labels > 0]
    excluded_labels: list[int] = []
    excluded_candidates: list[dict[str, float | int]] = []

    for label_value in candidate_labels:
        y_coordinates, x_coordinates = np.nonzero(labels == label_value)
        if x_coordinates.size == 0:
            continue

        pixel_distances = np.hypot(
            x_coordinates.astype(np.float64) - center_x,
            y_coordinates.astype(np.float64) - center_y,
        )
        centroid_x = float(x_coordinates.mean())
        centroid_y = float(y_coordinates.mean())
        centroid_distance = float(np.hypot(centroid_x - center_x, centroid_y - center_y))

        if boundary_policy == "overlap":
            should_exclude = bool(np.any(pixel_distances >= threshold_radius))
        elif boundary_policy == "centroid":
            should_exclude = centroid_distance >= threshold_radius
        else:
            # 黒点の全画素が境界円の外側にある場合だけ除外する。
            should_exclude = bool(np.all(pixel_distances >= threshold_radius))

        if should_exclude:
            label = int(label_value)
            excluded_labels.append(label)
            excluded_candidates.append(
                {
                    "label": label,
                    "centroid_distance_px": centroid_distance,
                    "minimum_pixel_distance_px": float(pixel_distances.min()),
                    "maximum_pixel_distance_px": float(pixel_distances.max()),
                }
            )

    masked_labels = labels.copy()
    if excluded_labels:
        masked_labels[np.isin(masked_labels, excluded_labels)] = 0

    count = max(0, int(original_count) - len(excluded_labels))
    LOGGER.info(
        "Rmask で除外した黒点数: %d (limit=%.3f px, boundary_policy=%s)",
        len(excluded_labels),
        threshold_radius,
        boundary_policy,
    )
    LOGGER.info("Rmask 適用後の黒点数: %d", count)

    masked_image = np.where(masked_labels > 0, 0, 255).astype(np.uint8)
    if not return_details:
        return count, masked_image

    details = {
        "enabled": True,
        "center_x_px": center_x,
        "center_y_px": center_y,
        "solar_radius_px": float(solar_radius),
        "radius_ratio": float(radius_ratio),
        "threshold_radius_px": threshold_radius,
        "boundary_policy": boundary_policy,
        "excluded_candidate_count": len(excluded_labels),
        "excluded_candidate_labels": sorted(excluded_labels),
        "excluded_candidates": excluded_candidates,
    }
    return count, masked_image, details
