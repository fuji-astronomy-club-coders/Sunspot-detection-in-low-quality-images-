"""輝度ヒストグラムを対数目盛で描画する Qt ウィジェット。"""

import numpy as np
from PyQt6.QtCore import QEvent, QRectF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPaintEvent, QPainter, QPen
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


def histogram_bin_index(
    position: float,
    left: float,
    width: float,
    bin_count: int,
) -> int | None:
    """カーソルのX座標からヒストグラムのビン番号を求める。

    Parameters
    ----------
    position : float
        ウィジェット上のカーソルX座標。
    left : float
        プロット領域の左端座標。
    width : float
        プロット領域の幅。
    bin_count : int
        ヒストグラムのビン数。

    Returns
    -------
    int | None
        対応するビン番号。プロット領域外または幅・ビン数が不正な場合はNone。
    """
    if width <= 0.0 or bin_count <= 0:
        return None
    relative_position = position - left
    if relative_position < 0.0 or relative_position > width:
        return None
    return min(int(relative_position / width * bin_count), bin_count - 1)


class BrightnessHistogramWidget(QWidget):
    """カーソル位置の輝度値と画素数を確認できる対数ヒストグラム。"""

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
        self._hovered_bin: int | None = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumHeight(132)
        self.setMaximumHeight(176)
        self.setToolTip(
            "グラフ上にカーソルを移動すると、その位置の輝度値と画素数を表示します。"
        )

    def _plot_rect(self) -> QRectF:
        """グラフの描画領域を返す。

        Returns
        -------
        QRectF
            軸ラベルを除いたヒストグラムのプロット領域。
        """
        return QRectF(
            64,
            27,
            max(1, self.width() - 88),
            max(1, self.height() - 61),
        )

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """カーソル位置に対応する輝度ビンを選択する。

        Parameters
        ----------
        event : QMouseEvent | None
            マウス移動イベント。
        """
        if event is None:
            return
        plot = self._plot_rect()
        position = event.position()
        if plot.top() <= position.y() <= plot.bottom():
            selected_bin = histogram_bin_index(
                position.x(),
                plot.left(),
                plot.width(),
                self._counts.size,
            )
        else:
            selected_bin = None

        if selected_bin != self._hovered_bin:
            self._hovered_bin = selected_bin
            self.update()

    def leaveEvent(self, event: QEvent | None) -> None:
        """カーソルがヒストグラムから離れたときに選択表示を解除する。

        Parameters
        ----------
        event : QEvent | None
            マウス離脱イベント。
        """
        self._hovered_bin = None
        self.update()
        super().leaveEvent(event)

    def _hover_text(self) -> str:
        """選択中のビンに対応する輝度範囲と画素数を整形する。

        Returns
        -------
        str
            ヒストグラム上部に表示するカーソル位置の情報。
        """
        if self._hovered_bin is None:
            return f"輝度ヒストグラム  |  対象画素: {self._pixel_count:,}"

        index = self._hovered_bin
        lower = float(self._edges[index])
        upper = float(self._edges[index + 1])
        center = (lower + upper) / 2.0
        count = int(self._counts[index])
        return (
            f"輝度: {center:.6g}  [{lower:.6g}, {upper:.6g})"
            f"  |  画素数: {count:,}  |  合計: {self._pixel_count:,}"
        )

    def paintEvent(self, event: QPaintEvent | None) -> None:
        """対数縦軸の棒グラフ、カーソルガイド、値ラベルを描画する。

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
            self._hover_text(),
        )

        plot = self._plot_rect()
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

            if self._hovered_bin is not None:
                index = self._hovered_bin
                marker_x = plot.left() + (index + 0.5) * bin_width
                guide_pen = QPen(QColor(255, 220, 120), 1.5, Qt.PenStyle.DashLine)
                guide_pen.setCosmetic(True)
                painter.setPen(guide_pen)
                painter.drawLine(
                    int(round(marker_x)),
                    int(plot.top()),
                    int(round(marker_x)),
                    int(plot.bottom()),
                )

                selected_height = float(self._relative_heights[index]) * plot.height()
                marker_y = plot.bottom() - max(1.0, selected_height)
                painter.setPen(QPen(QColor(255, 245, 190), 1.5))
                painter.setBrush(QColor(255, 190, 70))
                painter.drawEllipse(
                    QRectF(marker_x - 3.5, marker_y - 3.5, 7.0, 7.0)
                )

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
