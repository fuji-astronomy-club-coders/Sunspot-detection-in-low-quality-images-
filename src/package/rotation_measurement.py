"""Geometry helpers for aligning two solar images from matching landmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SimilarityAlignment:
    """A similarity transform mapping comparison-image pixels to reference pixels."""

    matrix: NDArray[np.float64]
    rotation_deg: float
    scale: float

    def transform_points(
        self,
        points: NDArray[np.floating[Any]] | list[tuple[float, float]],
    ) -> NDArray[np.float64]:
        """Transform an N-by-2 array of pixel coordinates."""
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("座標は N×2 の配列にしてください。")
        homogeneous = np.column_stack((values, np.ones(len(values))))
        return homogeneous @ self.matrix.T


def calculate_similarity_alignment(
    reference_points: NDArray[np.floating[Any]] | list[tuple[float, float]],
    comparison_points: NDArray[np.floating[Any]] | list[tuple[float, float]],
) -> SimilarityAlignment:
    """Calculate rotation, scale and translation from two matching point pairs.

    The matrix maps comparison-image coordinates to reference-image coordinates.
    With image coordinates (x right, y down), positive angles are clockwise.
    """
    reference = np.asarray(reference_points, dtype=np.float64)
    comparison = np.asarray(comparison_points, dtype=np.float64)
    if reference.shape != (2, 2) or comparison.shape != (2, 2):
        raise ValueError("各画像で対応点を2点ずつ指定してください。")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(comparison)):
        raise ValueError("対応点の座標は有限値にしてください。")

    reference_vector = reference[1] - reference[0]
    comparison_vector = comparison[1] - comparison[0]
    reference_length = float(np.linalg.norm(reference_vector))
    comparison_length = float(np.linalg.norm(comparison_vector))
    if reference_length <= 1e-9 or comparison_length <= 1e-9:
        raise ValueError("2点は異なる位置を選択してください。")

    reference_angle = math.atan2(float(reference_vector[1]), float(reference_vector[0]))
    comparison_angle = math.atan2(float(comparison_vector[1]), float(comparison_vector[0]))
    angle = reference_angle - comparison_angle
    scale = reference_length / comparison_length
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    linear = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = reference[0] - linear @ comparison[0]
    matrix = np.column_stack((linear, translation))
    rotation_deg = (math.degrees(angle) + 180.0) % 360.0 - 180.0
    return SimilarityAlignment(matrix=matrix, rotation_deg=rotation_deg, scale=scale)


def centered_fit_matrix(
    reference_shape: tuple[int, int],
    comparison_shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Create a non-distorting initial fit that centers the comparison image."""
    reference_height, reference_width = reference_shape
    comparison_height, comparison_width = comparison_shape
    if min(reference_height, reference_width, comparison_height, comparison_width) <= 0:
        raise ValueError("画像サイズは正の値にしてください。")
    scale = min(
        reference_width / comparison_width,
        reference_height / comparison_height,
    )
    offset_x = (reference_width - comparison_width * scale) / 2.0
    offset_y = (reference_height - comparison_height * scale) / 2.0
    return np.asarray(
        [[scale, 0.0, offset_x], [0.0, scale, offset_y]],
        dtype=np.float64,
    )


def add_rotation_about_center(
    matrix: NDArray[np.floating[Any]],
    angle_deg: float,
    center: tuple[float, float],
) -> NDArray[np.float64]:
    """Compose an image-coordinate rotation around a point after an existing transform."""
    source_matrix = np.asarray(matrix, dtype=np.float64)
    if source_matrix.shape != (2, 3):
        raise ValueError("変換行列は 2×3 にしてください。")
    if not math.isfinite(angle_deg):
        raise ValueError("回転角は有限値にしてください。")
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    center_x, center_y = center
    rotation = np.asarray(
        [
            [cosine, -sine, center_x - cosine * center_x + sine * center_y],
            [sine, cosine, center_y - sine * center_x - cosine * center_y],
        ],
        dtype=np.float64,
    )
    homogeneous = np.vstack((source_matrix, [0.0, 0.0, 1.0]))
    rotation_homogeneous = np.vstack((rotation, [0.0, 0.0, 1.0]))
    return (rotation_homogeneous @ homogeneous)[:2]
