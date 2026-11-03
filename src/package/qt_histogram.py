"""輝度ヒストグラムを対数目盛で描画する Qt ウィジェット。"""

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPaintEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget


def calculate_brightness_histogram(
    image: np.ndarray,
    bins: int = 256,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """有限な画素値から輝度ヒストグラムを計算する。

    Parameters
    ----------
    image : numpy.ndarray
        2次元の輝度画像。
    bins : int, optional
        ヒストグラムのビン数。

    Returns
    -------
    tuple
        画素数、ビン境界、最小輝度、最大輝度、対象画素数。

    Raises
    ------
    ValueError
        入力画像またはビン数が不正な場合。
    """
    if image.ndim != 2 or image.size == 0:
        raise ValueError("ヒストグラム画像は空でない2次元配列にしてください。")
    if bins < 1:
        raise ValueError("ヒストグラムのビン数は1以上にしてください。")

    values = np.asarray(image, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        edges = np.linspace(0.0, 1.0, bins + 1)
        return np.zeros(bins, dtype=np.int64), edges, 0.0, 0.0, 0

    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    if maximum == minimum:
        edges = np.linspace(minimum - 0.5, maximum + 0.5, bins + 1)
        counts = np.zeros(bins, dtype=np.int64)
        counts[bins // 2] = finite_values.size
    else:
        counts, edges = np.histogram(
            finite_values,
            bins=bins,
            range=(minimum, maximum),
        )
        counts = counts.astype(np.int64, copy=False)

    return counts, edges, minimum, maximum, int(finite_values.size)


def log_scaled_counts(counts: np.ndarray) -> np.ndarray:
    """画素数をlog10(count + 1)で0～1へ正規化する。

    Parameters
    ----------
    counts : numpy.ndarray
        ビンごとの画素数。

    Returns
    -------
    numpy.ndarray
        対数変換後の相対高さ。最大値は1。
    """
    values = np.asarray(counts, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("ヒストグラムの画素数は0以上にしてください。")
    transformed = np.log10(values + 1.0)
    maximum = float(transformed.max(initial=0.0))
    if maximum <= 0.0:
        return np.zeros(values.shape, dtype=np.float64)
    return transformed / maximum


class BrightnessHistogramWidget(QWidget):
    """輝度分布を対数縦軸で表示する軽量なQPainterウィジェット。"""

    def __init__(
        self,
        image: np.ndarray,
        parent: QWidget | None = None,
    ) -> None:
        """表示対象画像の輝度ヒストグラムを初期化する。

        Parameters
        ----------
        image : numpy.ndarray
            ヒストグラムの対象となる2次元輝度画像。
        parent : QWidget | None, optional
            親ウィジェット。
        """
        super().__init__(parent)
        (
            self._counts,
            self._edges,
            self._minimum,
            self._maximum,
            self._pixel_count,
        ) = calculate_brightness_histogram(image)
        self._relative_heights = log_scaled_counts(self._counts)
        self.setMinimumHeight(132)
        self.setMaximumHeight(176)
        self.setToolTip(
            "横軸: 輝度値  |  縦軸: 画素数（log10(count + 1)）"
        )

    def paintEvent(self, event: QPaintEvent | None) -> None:
        """対数縦軸の棒グラフ、目盛り、輝度範囲を描画する。

        Parameters
        ----------
        event : QPaintEvent | None
            Qtの描画イベント。
        """
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        title_color = QColor(235, 235, 235)
        muted_color = QColor(185, 185, 185)
        grid_color = QColor(76, 76, 76)
        painter.setPen(title_color)
        painter.drawText(
            QRectF(8, 2, self.width() - 16, 20),
            Qt.AlignmentFlag.AlignCenter,
            f"輝度ヒストグラム  |  対象画素: {self._pixel_count:,}",
        )

        plot = QRectF(
            64,
            27,
            max(1, self.width() - 88),
            max(1, self.height() - 61),
        )
        maximum_count = int(self._counts.max(initial=0))
        maximum_log = float(np.log10(maximum_count + 1.0))
        if maximum_log <= 0.0:
            maximum_log = 1.0

        tick_values = [1]
        while tick_values[-1] * 10 < maximum_count:
            tick_values.append(tick_values[-1] * 10)
        if maximum_count > 1 and maximum_count not in tick_values:
            tick_values.append(maximum_count)

        for tick in tick_values:
            y = plot.bottom() - (
                np.log10(tick + 1.0) / maximum_log
            ) * plot.height()
            painter.setPen(QPen(grid_color, 1, Qt.PenStyle.DotLine))
            painter.drawLine(
                int(plot.left()),
                round(y),
                int(plot.right()),
                round(y),
            )
            painter.setPen(muted_color)
            painter.drawText(
                QRectF(2, y - 9, 55, 18),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{tick:,}",
            )

        painter.setPen(QPen(muted_color, 1))
        painter.drawLine(
            int(plot.left()),
            int(plot.top()),
            int(plot.left()),
            int(plot.bottom()),
        )
        painter.drawLine(
            int(plot.left()),
            int(plot.bottom()),
            int(plot.right()),
            int(plot.bottom()),
        )

        if self._counts.size:
            bin_width = plot.width() / self._counts.size
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(110, 175, 220))
            for index, relative_height in enumerate(self._relative_heights):
                if relative_height <= 0.0:
                    continue
                bar_height = max(1.0, float(relative_height) * plot.height())
                bar = QRectF(
                    plot.left() + index * bin_width,
                    plot.bottom() - bar_height,
                    max(1.0, bin_width - 0.35),
                    bar_height,
                )
                painter.drawRect(bar)

        painter.setPen(muted_color)
        if self._minimum == self._maximum:
            center = plot.center().x()
            painter.drawText(
                QRectF(center - 65, plot.bottom() + 4, 130, 18),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                f"{self._minimum:g}",
            )
        else:
            labels = (
                (plot.left(), f"{self._minimum:g}", Qt.AlignmentFlag.AlignLeft),
                (
                    plot.center().x(),
                    f"{(self._minimum + self._maximum) / 2:g}",
                    Qt.AlignmentFlag.AlignHCenter,
                ),
                (plot.right(), f"{self._maximum:g}", Qt.AlignmentFlag.AlignRight),
            )
            for x, label, alignment in labels:
                painter.drawText(
                    QRectF(x - 55, plot.bottom() + 4, 110, 18),
                    alignment | Qt.AlignmentFlag.AlignTop,
                    label,
                )

        painter.drawText(
            QRectF(plot.left(), self.height() - 19, plot.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            "輝度値",
        )
        painter.save()
        painter.translate(13, plot.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-plot.height() / 2, -10, plot.height(), 18),
            Qt.AlignmentFlag.AlignCenter,
            "画素数（対数目盛）",
        )
        painter.restore()
        painter.end()
