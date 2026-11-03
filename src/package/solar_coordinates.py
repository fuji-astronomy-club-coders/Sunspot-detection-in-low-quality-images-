"""Solar-disk normalized coordinate transforms for multi-observatory comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SolarGeometry:
    """MIN2 geometry for one solar image."""

    center: tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        center_x, center_y = self.center
        if not np.isfinite(center_x) or not np.isfinite(center_y):
            raise ValueError("太陽中心は有限値である必要があります。")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("太陽半径は正の有限値である必要があります。")


@dataclass(frozen=True, slots=True)
class Orientation:
    """Solar north in Cartesian image-plane coordinates (+X right, +Y up)."""

    north_vector: tuple[float, float]
    flip_x: bool = False

    def __post_init__(self) -> None:
        north_x, north_y = self.north_vector
        if not np.isfinite(north_x) or not np.isfinite(north_y):
            raise ValueError("north_vector は有限値である必要があります。")
        if np.hypot(north_x, north_y) == 0:
            raise ValueError("north_vector にゼロベクトルは指定できません。")

    @property
    def north_angle_deg(self) -> float:
        """Return the Cartesian polar angle from +X, normalized to [0, 360)."""
        angle = float(np.degrees(np.arctan2(self.north_vector[1], self.north_vector[0])) % 360.0)
        return 0.0 if np.isclose(angle, 360.0) or np.isclose(angle, 0.0) else angle


@dataclass(frozen=True, slots=True)
class SolarTransform:
    """Bidirectional transform between image pixels and normalized solar coordinates."""

    geometry: SolarGeometry
    orientation: Orientation

    @property
    def _basis(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # north_vector is expressed in a Cartesian image-plane basis:
        # +X points right and +Y points up. Pixel coordinates have +Y down,
        # so convert the vector before constructing the image-space basis.
        north_cartesian = np.asarray(self.orientation.north_vector, dtype=np.float64)
        north_cartesian /= np.linalg.norm(north_cartesian)
        north = np.array(
            [north_cartesian[0], -north_cartesian[1]],
            dtype=np.float64,
        )
        positive_x = np.array([-north[1], north[0]], dtype=np.float64)
        if self.orientation.flip_x:
            positive_x *= -1.0
        return positive_x, north

    def pixel_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        """Convert one zero-based pixel coordinate to solar-radius units."""
        positive_x, north = self._basis
        displacement = np.asarray(
            [x - self.geometry.center[0], y - self.geometry.center[1]],
            dtype=np.float64,
        ) / self.geometry.radius
        return float(np.dot(displacement, positive_x)), float(np.dot(displacement, north))

    def normalized_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Convert one normalized solar coordinate to image pixels."""
        positive_x, north = self._basis
        displacement = (
            np.asarray([x, y], dtype=np.float64)[0] * positive_x
            + np.asarray([x, y], dtype=np.float64)[1] * north
        ) * self.geometry.radius
        return (
            float(self.geometry.center[0] + displacement[0]),
            float(self.geometry.center[1] + displacement[1]),
        )

    def pixel_points_to_normalized(
        self,
        points: list[tuple[float, float]] | NDArray[np.floating[Any]],
    ) -> list[list[float]]:
        """Convert a sequence of image-space points to normalized coordinates."""
        return [list(self.pixel_to_normalized(float(x), float(y))) for x, y in points]

    def normalized_points_to_pixel(
        self,
        points: list[list[float]] | list[tuple[float, float]],
    ) -> NDArray[np.float64]:
        """Convert normalized polygon vertices to image-space coordinates."""
        return np.asarray(
            [self.normalized_to_pixel(float(x), float(y)) for x, y in points],
            dtype=np.float64,
        )

    def rectangle_to_normalized_polygon(
        self,
        rectangle: tuple[float, float, float, float],
    ) -> list[list[float]]:
        """Convert an image-space rectangle to a common-coordinate quadrilateral."""
        x, y, width, height = rectangle
        # Pixel coordinates identify pixel centers. Use half-pixel outer edges
        # so a selected width-by-height ROI maps to exactly those pixel cells.
        left, top = x - 0.5, y - 0.5
        right, bottom = x + width - 0.5, y + height - 0.5
        corners = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        ]
        return self.pixel_points_to_normalized(corners)

    def as_dict(self) -> dict[str, Any]:
        """Return the effective transform inputs for reproducibility logs."""
        positive_x, north = self._basis
        center_x, center_y = self.geometry.center
        radius = self.geometry.radius
        matrix = [
            [float(positive_x[0] / radius), float(positive_x[1] / radius)],
            [float(north[0] / radius), float(north[1] / radius)],
        ]
        offset = [
            float(-(positive_x[0] * center_x + positive_x[1] * center_y) / radius),
            float(-(north[0] * center_x + north[1] * center_y) / radius),
        ]
        return {
            "coordinate_system": "solar_disk_normalized",
            "direction_coordinate_system": "cartesian_xy",
            "north_angle_deg": self.orientation.north_angle_deg,
            "positive_x": "right",
            "positive_y": "up",
            "matrix": matrix,
            "offset": offset,
            "center_x_px": float(center_x),
            "center_y_px": float(center_y),
            "solar_radius_px": float(radius),
            "north_vector": {
                "x": float(self.orientation.north_vector[0]),
                "y": float(self.orientation.north_vector[1]),
            },
            "flip_x": bool(self.orientation.flip_x),
        }
