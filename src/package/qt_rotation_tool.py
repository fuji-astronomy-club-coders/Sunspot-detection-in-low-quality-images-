"""PyQt GUI for registering and measuring the relative rotation of solar images."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.package.qt_dialogs import ensure_qapplication
from src.package.qt_image import colorize_intensity
from src.package.rotation_measurement import (
    add_rotation_about_center,
    calculate_similarity_alignment,
    centered_fit_matrix,
)
from src.package.utils import loadImage

IMAGE_FILTER = "Solar images (*.tif *.tiff *.fits *.fit *.fts);;All files (*)"


def _rgb_image(image: np.ndarray) -> np.ndarray:
    """Convert a 2D scientific image into an independent RGB preview."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("空でない2次元のグレースケール画像を指定してください。")
    bgr, _minimum, _maximum = colorize_intensity(image, "GRAY")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _qimage(image: np.ndarray) -> QImage:
    """Create an owned Qt image from an RGB array."""
    values = np.ascontiguousarray(image, dtype=np.uint8)
    return QImage(
        values.data,
        values.shape[1],
        values.shape[0],
        int(values.strides[0]),
        QImage.Format.Format_RGB888,
    ).copy()


class _ImageCanvas(QWidget):
    """Aspect-ratio preserving image view with numbered point selection."""

    point_clicked = pyqtSignal(float, float)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.image: QImage | None = None
        self.image_width = 0
        self.image_height = 0
        self.points: list[tuple[float, float]] = []
        self.setMinimumSize(320, 280)
        self.setMouseTracking(True)

    def set_image(self, image: np.ndarray | None) -> None:
        """Replace the image displayed by the canvas."""
        self.image = _qimage(image) if image is not None else None
        self.image_width = self.image.width() if self.image else 0
        self.image_height = self.image.height() if self.image else 0
        self.points.clear()
        self.update()

    def _image_rect(self) -> QRectF:
        """Return the centered image rectangle in widget coordinates."""
        if self.image is None:
            return QRectF()
        scale = min(
            max(1, self.width() - 24) / self.image_width,
            max(1, self.height() - 44) / self.image_height,
        )
        width = self.image_width * scale
        height = self.image_height * scale
        return QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2 + 10,
            width,
            height,
        )

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802
        """Draw the image and numbered selection markers."""
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(27, 31, 38))
        painter.setPen(QColor(235, 239, 244))
        painter.drawText(10, 20, self.title)
        if self.image is None:
            painter.setPen(QColor(155, 163, 175))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "画像を読み込んでください",
            )
            return
        rect = self._image_rect()
        painter.drawImage(rect, self.image)
        painter.setPen(QPen(QColor(80, 88, 100), 1))
        painter.drawRect(rect)
        for index, (x, y) in enumerate(self.points, start=1):
            marker = QPointF(
                rect.left() + x * rect.width() / self.image_width,
                rect.top() + y * rect.height() / self.image_height,
            )
            painter.setPen(QPen(QColor(255, 220, 70), 2))
            painter.drawEllipse(marker, 7, 7)
            painter.setPen(QColor(255, 245, 180))
            painter.drawText(marker + QPointF(9, -8), str(index))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Convert a left click into an original-image pixel coordinate."""
        if self.image is None or event.button() != Qt.MouseButton.LeftButton:
            return
        rect = self._image_rect()
        position = event.position()
        if not rect.contains(position):
            return
        x = (position.x() - rect.left()) * self.image_width / rect.width()
        y = (position.y() - rect.top()) * self.image_height / rect.height()
        self.point_clicked.emit(float(x), float(y))


class SolarRotationToolDialog(QDialog):
    """Interactive two-image registration, overlay and relative-angle tool."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("太陽画像の回転角測定")
        self.resize(1350, 850)
        self.reference_path: Path | None = None
        self.comparison_path: Path | None = None
        self.reference: np.ndarray | None = None
        self.comparison: np.ndarray | None = None
        self.reference_rgb: np.ndarray | None = None
        self.comparison_rgb: np.ndarray | None = None
        self.reference_points: list[tuple[float, float]] = []
        self.comparison_points: list[tuple[float, float]] = []
        self.base_matrix: np.ndarray | None = None
        self.automatic_angle = 0.0
        self.scale = 1.0
        self.manual_angle = 0.0
        self.opacity = 50

        root = QVBoxLayout(self)
        file_row = QHBoxLayout()
        self.reference_button = QPushButton("画像 A を開く")
        self.comparison_button = QPushButton("画像 B を開く")
        self.reference_name = QLabel("画像 A: 未選択")
        self.comparison_name = QLabel("画像 B: 未選択")
        self.reference_button.clicked.connect(self._open_reference)
        self.comparison_button.clicked.connect(self._open_comparison)
        file_row.addWidget(self.reference_button)
        file_row.addWidget(self.reference_name, 1)
        file_row.addSpacing(12)
        file_row.addWidget(self.comparison_button)
        file_row.addWidget(self.comparison_name, 1)
        root.addLayout(file_row)

        canvases = QHBoxLayout()
        self.reference_canvas = _ImageCanvas("A: 基準画像")
        self.comparison_canvas = _ImageCanvas("B: 比較画像")
        self.reference_canvas.point_clicked.connect(self._add_reference_point)
        self.comparison_canvas.point_clicked.connect(self._add_comparison_point)
        canvases.addWidget(self.reference_canvas, 1)
        canvases.addWidget(self.comparison_canvas, 1)
        root.addLayout(canvases, 1)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("B の透過率"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(self.opacity)
        self.opacity_label = QLabel("50%")
        self.opacity_slider.valueChanged.connect(self._set_opacity)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_label)
        root.addLayout(opacity_row)

        adjustment_row = QHBoxLayout()
        adjustment_row.addWidget(QLabel("追加回転（画像座標・時計回り +）"))
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setDecimals(2)
        self.rotation_spin.setSingleStep(0.1)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.valueChanged.connect(self._set_manual_rotation)
        adjustment_row.addWidget(self.rotation_spin)
        self.calculate_button = QPushButton("4点から自動位置合わせ")
        self.clear_button = QPushButton("対応点をクリア")
        self.reset_button = QPushButton("初期配置に戻す")
        self.save_button = QPushButton("重ね合わせ画像・測定値を保存")
        self.calculate_button.clicked.connect(self._calculate_alignment)
        self.clear_button.clicked.connect(self._clear_points)
        self.reset_button.clicked.connect(self._reset_alignment)
        self.save_button.clicked.connect(self._save_results)
        adjustment_row.addWidget(self.calculate_button)
        adjustment_row.addWidget(self.clear_button)
        adjustment_row.addWidget(self.reset_button)
        adjustment_row.addWidget(self.save_button)
        root.addLayout(adjustment_row)

        self.overlay_canvas = _ImageCanvas("A + B: 透過重ね合わせ")
        self.measurement_label = QLabel(
            "画像 A・B を読み込み、各画像で同じ特徴点を2点ずつクリックしてください。"
        )
        self.measurement_label.setWordWrap(True)
        root.addWidget(self.overlay_canvas, 2)
        root.addWidget(self.measurement_label)

    def _load(self, path: str, reference: bool) -> None:
        """Load and validate one TIFF/FITS grayscale image."""
        try:
            image = np.asarray(loadImage(path))
            rgb = _rgb_image(image)
        except Exception as error:
            QMessageBox.critical(self, "画像を読み込めません", f"{path}\n\n{error}")
            return
        if reference:
            self.reference_path, self.reference, self.reference_rgb = Path(path), image, rgb
            self.reference_name.setText(str(path))
            self.reference_canvas.set_image(rgb)
        else:
            self.comparison_path, self.comparison, self.comparison_rgb = Path(path), image, rgb
            self.comparison_name.setText(str(path))
            self.comparison_canvas.set_image(rgb)
        self._clear_points()
        self._reset_alignment(show_message=False)

    def _open_reference(self) -> None:
        """Open the reference-image file picker."""
        path, _ = QFileDialog.getOpenFileName(
            self, "基準画像 A を選択", "", IMAGE_FILTER
        )
        if path:
            self._load(path, True)

    def _open_comparison(self) -> None:
        """Open the comparison-image file picker."""
        path, _ = QFileDialog.getOpenFileName(
            self, "比較画像 B を選択", "", IMAGE_FILTER
        )
        if path:
            self._load(path, False)

    def _add_reference_point(self, x: float, y: float) -> None:
        """Append a reference-image landmark, keeping at most two."""
        if len(self.reference_points) >= 2:
            self.reference_points.clear()
        self.reference_points.append((x, y))
        self.reference_canvas.points = self.reference_points.copy()
        self.reference_canvas.update()
        self._update_measurement()

    def _add_comparison_point(self, x: float, y: float) -> None:
        """Append a comparison-image landmark, keeping at most two."""
        if len(self.comparison_points) >= 2:
            self.comparison_points.clear()
        self.comparison_points.append((x, y))
        self.comparison_canvas.points = self.comparison_points.copy()
        self.comparison_canvas.update()
        self._update_measurement()

    def _calculate_alignment(self) -> None:
        """Calculate rotation, uniform scale and translation from landmarks."""
        if self.reference_rgb is None or self.comparison_rgb is None:
            QMessageBox.warning(self, "画像未選択", "画像 A と画像 B を読み込んでください。")
            return
        try:
            result = calculate_similarity_alignment(
                self.reference_points, self.comparison_points
            )
        except ValueError as error:
            QMessageBox.warning(self, "対応点が不足しています", str(error))
            return
        self.base_matrix = result.matrix
        self.automatic_angle = result.rotation_deg
        self.scale = result.scale
        self.rotation_spin.blockSignals(True)
        self.rotation_spin.setValue(0.0)
        self.rotation_spin.blockSignals(False)
        self.manual_angle = 0.0
        self._refresh_overlay()
        self._update_measurement()

    def _clear_points(self) -> None:
        """Clear both sets of landmark clicks without changing the image alignment."""
        self.reference_points.clear()
        self.comparison_points.clear()
        self.reference_canvas.points = []
        self.comparison_canvas.points = []
        self.reference_canvas.update()
        self.comparison_canvas.update()
        self._update_measurement()

    def _reset_alignment(self, show_message: bool = True) -> None:
        """Return to a centered, aspect-ratio-preserving initial placement."""
        self.automatic_angle = 0.0
        self.manual_angle = 0.0
        self.rotation_spin.blockSignals(True)
        self.rotation_spin.setValue(0.0)
        self.rotation_spin.blockSignals(False)
        if self.reference_rgb is not None and self.comparison_rgb is not None:
            self.base_matrix = centered_fit_matrix(
                self.reference_rgb.shape[:2], self.comparison_rgb.shape[:2]
            )
            self.scale = float(math.sqrt(abs(np.linalg.det(self.base_matrix[:, :2]))))
        else:
            self.base_matrix = None
            self.scale = 1.0
        self._refresh_overlay()
        self._update_measurement()
        if show_message:
            self.measurement_label.setText(
                self.measurement_label.text() + "\n初期配置に戻しました。"
            )

    def _set_manual_rotation(self, value: float) -> None:
        """Apply an additional image-coordinate rotation to image B."""
        self.manual_angle = float(value)
        self._refresh_overlay()
        self._update_measurement()

    def _set_opacity(self, value: int) -> None:
        """Update the overlay transparency."""
        self.opacity = int(value)
        self.opacity_label.setText(f"{value}%")
        self._refresh_overlay()

    def _current_matrix(self) -> np.ndarray | None:
        """Return the full comparison-to-reference image transform."""
        if self.base_matrix is None or self.reference_rgb is None:
            return None
        height, width = self.reference_rgb.shape[:2]
        return add_rotation_about_center(
            self.base_matrix, self.manual_angle, (width / 2.0, height / 2.0)
        )

    def _warped_comparison(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Warp comparison RGB pixels and a validity mask into reference space."""
        if self.reference_rgb is None or self.comparison_rgb is None:
            return None
        matrix = self._current_matrix()
        if matrix is None:
            return None
        height, width = self.reference_rgb.shape[:2]
        warped = cv2.warpAffine(
            self.comparison_rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        source_mask = np.full(self.comparison_rgb.shape[:2], 255, dtype=np.uint8)
        valid = cv2.warpAffine(
            source_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.ascontiguousarray(warped), np.ascontiguousarray(valid)

    def _composite(self) -> np.ndarray | None:
        """Blend image B over A while preserving pixels outside B's footprint."""
        if self.reference_rgb is None:
            return None
        warped_result = self._warped_comparison()
        if warped_result is None:
            return self.reference_rgb
        warped, valid = warped_result
        alpha = (valid.astype(np.float32) / 255.0) * (self.opacity / 100.0)
        blend = self.reference_rgb.astype(np.float32) * (1.0 - alpha[..., None])
        blend += warped.astype(np.float32) * alpha[..., None]
        return np.clip(blend, 0, 255).astype(np.uint8)

    def _refresh_overlay(self) -> None:
        """Refresh the live overlay preview."""
        self.overlay_canvas.set_image(self._composite())

    def _update_measurement(self) -> None:
        """Update landmark status and derived rotation values."""
        point_status = (
            f"A: {len(self.reference_points)}/2 点　 B: {len(self.comparison_points)}/2 点"
        )
        total_correction = self.automatic_angle + self.manual_angle
        alignment_status = (
            f"自動補正角: {self.automatic_angle:+.3f}° / 倍率: {self.scale:.6f}"
            if self.base_matrix is not None
            else "位置合わせ: 中央・アスペクト比を保持して初期配置"
        )
        self.measurement_label.setText(
            f"{point_status}\n{alignment_status}\n"
            f"B に適用した総補正角: {total_correction:+.3f}°\n"
            f"A に対する B の相対回転角: {-total_correction:+.3f}°"
        )

    def _save_results(self) -> None:
        """Save the blended PNG and a JSON record of the measurement."""
        composite = self._composite()
        matrix = self._current_matrix()
        if composite is None or matrix is None or self.comparison_rgb is None:
            QMessageBox.warning(
                self, "保存できません", "画像を2枚読み込み、位置合わせを行ってください。"
            )
            return
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "重ね合わせ画像を保存",
            "sunspot_overlay.png",
            "PNG image (*.png)",
        )
        if not path_text:
            return
        output_path = Path(path_text)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        if not cv2.imwrite(
            str(output_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        ):
            QMessageBox.critical(
                self, "保存失敗", f"PNG を保存できませんでした: {output_path}"
            )
            return

        total_correction = self.automatic_angle + self.manual_angle
        record: dict[str, Any] = {
            "reference_image": str(self.reference_path) if self.reference_path else None,
            "comparison_image": str(self.comparison_path) if self.comparison_path else None,
            "coordinate_convention": (
                "pixel x right, y down; positive rotation is clockwise on screen"
            ),
            "reference_points_px": self.reference_points,
            "comparison_points_px": self.comparison_points,
            "automatic_correction_angle_deg": self.automatic_angle,
            "manual_additional_angle_deg": self.manual_angle,
            "total_correction_angle_deg": total_correction,
            "comparison_relative_rotation_deg": -total_correction,
            "comparison_to_reference_scale": self.scale,
            "comparison_to_reference_matrix": matrix.tolist(),
            "overlay_opacity_percent": self.opacity,
            "overlay_image": str(output_path),
        }
        json_path = output_path.with_suffix(".json")
        try:
            json_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            QMessageBox.warning(self, "JSON 保存失敗", str(error))
            return
        QMessageBox.information(
            self, "保存しました", f"画像: {output_path}\n測定値: {json_path}"
        )


def run_rotation_tool() -> int:
    """Launch the standalone solar-image rotation measurement GUI."""
    application: QApplication = ensure_qapplication()
    application.setQuitOnLastWindowClosed(True)
    dialog = SolarRotationToolDialog()
    dialog.show()
    return application.exec()
