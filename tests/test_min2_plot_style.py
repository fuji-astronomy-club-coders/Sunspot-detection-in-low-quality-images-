"""Regression tests for configurable MIN2 scatter point styling."""

import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.axes import Axes

from src.module.MIN2ver2 import show_circle, show_circle_simple


class Min2PlotPointStyleTests(unittest.TestCase):
    """Ensure the detailed and compact MIN2 plots apply the configured marker style."""

    def test_detailed_plot_uses_configured_point_style(self) -> None:
        with (
            patch("matplotlib.pyplot.show"),
            patch.object(Axes, "scatter") as scatter_mock,
        ):
            show_circle(
                img_inst=np.zeros((32, 32), dtype=np.uint8),
                spots=[[8, 9, 1.0, 0]],
                cir_stat=False,
                n=2,
                point_size=72,
                point_color="#00aaff",
            )
            self.assertEqual(scatter_mock.call_args.kwargs["s"], 72)
            self.assertEqual(scatter_mock.call_args.kwargs["color"], "#00aaff")

    def test_simple_plot_uses_configured_point_style(self) -> None:
        with (
            patch("matplotlib.pyplot.show"),
            patch.object(Axes, "scatter") as scatter_mock,
        ):
            show_circle_simple(
                img_inst=np.zeros((32, 32), dtype=np.uint8),
                spots=[[8, 9, 1.0, 0]],
                cir_stat=False,
                n=2,
                markersize=72,
                point_color="#00aaff",
            )
            self.assertEqual(scatter_mock.call_args.kwargs["s"], 72)
            self.assertEqual(scatter_mock.call_args.kwargs["color"], "#00aaff")


if __name__ == "__main__":
    unittest.main()
