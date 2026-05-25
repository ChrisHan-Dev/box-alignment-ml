"""
Operational alignment geometry for warehouse carton OBB detection.

This module measures whether a carton's front face is parallel to the
shelf/conveyor direction — NOT whether the image quadrilateral matches exact
perspective projection.

Perspective skew vs operational misalignment
------------------------------------------
A box can look like a trapezoid in camera space while still being physically
aligned on the belt. Conversely, a box rotated on the shelf may still have a
near-horizontal bottom edge due to camera angle. We therefore:

  - Use a shelf/conveyor reference frame (not raw image x-axis alone).
  - Measure the front-face width axis (shelf-parallel edge pair), not the
    lowest polygon edge, which is sensitive to perspective foreshortening.
  - Label and evaluate against *desired operational orientation*, not
    photogrammetric corner positions.

Robotics alignment differs from document/OCR tasks where exact quadrilateral
fit matters. Here the OBB encodes semantic pose: "is this box square to the
rails?"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Default tolerance for teleoperation / alignment alarm (degrees)
DEFAULT_ALIGNMENT_THRESHOLD_DEG = 6.0


class ReferenceMode(str, Enum):
    """How the shelf/conveyor 'horizontal' reference is determined."""

    AUTO_CONVEYOR = "auto_conveyor"  # Hough lines on lower belt ROI (default)
    IMAGE_HORIZONTAL = "image_horizontal"  # image x-axis (0°)
    FIXED = "fixed"  # use reference_angle_deg from config


@dataclass(frozen=True)
class AlignmentConfig:
    """
    Operational reference frame for alignment measurement.

    Attributes:
        mode: How to obtain the shelf/conveyor direction.
        reference_angle_deg: Used when mode is FIXED; ignored otherwise.
        threshold_deg: Max deviation (degrees) still considered ALIGNED.
        conveyor_roi_start: Fraction of image height where belt ROI begins.
    """

    mode: ReferenceMode = ReferenceMode.AUTO_CONVEYOR
    reference_angle_deg: float = 0.0
    threshold_deg: float = DEFAULT_ALIGNMENT_THRESHOLD_DEG
    conveyor_roi_start: float = 0.55


@dataclass(frozen=True)
class OperationalAlignment:
    """Alignment metrics for one detected OBB polygon."""

    face_angle_deg: float  # shelf-parallel front-face axis (degrees)
    reference_angle_deg: float  # shelf/conveyor direction used as "horizontal"
    deviation_deg: float  # |face_angle - reference|, mod 90° symmetry
    status: str  # "ALIGNED" | "MISALIGNED"


def normalize_angle_deg(deg: float) -> float:
    """Map any angle to [-90, 90) for undirected line-direction comparison."""
    while deg >= 90.0:
        deg -= 180.0
    while deg < -90.0:
        deg += 180.0
    return deg


def estimate_conveyor_angle(
    image_bgr: np.ndarray,
    roi_start: float = 0.55,
) -> float:
    """
    Estimate conveyor rail / roller direction from the lower belt ROI.

    Returns the dominant near-horizontal direction of shelf rails in image
    space. This defines operational "horizontal" for alignment measurement.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    roi = cv2.GaussianBlur(gray[int(height * roi_start) :, :], (5, 5), 0)
    edges = cv2.Canny(roi, 30, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 35, minLineLength=max(width // 6, 40), maxLineGap=20
    )
    if lines is None:
        return 0.0

    sum_x = sum_y = total_weight = 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = normalize_angle_deg(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if abs(angle) < 35.0 and length > width // 8:
            rad = math.radians(angle)
            sum_x += length * math.cos(rad)
            sum_y += length * math.sin(rad)
            total_weight += length

    if total_weight == 0.0:
        return 0.0
    return normalize_angle_deg(math.degrees(math.atan2(sum_y, sum_x)))


def resolve_reference_angle(
    image_bgr: Optional[np.ndarray],
    config: AlignmentConfig,
) -> float:
    """Resolve the operational shelf/conveyor reference angle for one frame."""
    if config.mode == ReferenceMode.FIXED:
        return normalize_angle_deg(config.reference_angle_deg)
    if config.mode == ReferenceMode.IMAGE_HORIZONTAL:
        return 0.0
    if image_bgr is not None:
        return estimate_conveyor_angle(image_bgr, roi_start=config.conveyor_roi_start)
    return 0.0


def _polygon_edges(polygon: np.ndarray) -> List[Tuple[float, float]]:
    """Return (angle_deg, length) for each polygon edge."""
    edges: List[Tuple[float, float]] = []
    for idx in range(4):
        p1 = polygon[idx]
        p2 = polygon[(idx + 1) % 4]
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        length = math.hypot(dx, dy)
        angle = normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
        edges.append((angle, length))
    return edges


def _raw_angular_offset(line_angle: float, reference_angle: float) -> float:
    """Absolute difference between two line directions (no 90° fold)."""
    return abs(normalize_angle_deg(line_angle - reference_angle))


def deviation_from_reference(line_angle: float, reference_angle: float) -> float:
    """
    Operational deviation of a shelf-parallel front-face edge from the reference.

    Uses direct angular offset only. Do not fold by 90° here — that would treat
    the front-face height axis as equally "aligned" as the width axis.
    """
    return _raw_angular_offset(line_angle, reference_angle)


def shelf_parallel_face_angle(
    polygon: np.ndarray,
    reference_angle: float,
) -> Tuple[float, float]:
    """
    Angle of the front-face width axis — the edge pair parallel to the shelf.

    An OBB quadrilateral has two parallel pairs (edges 0&2 and 1&3). The pair
    whose direction is closest to the shelf reference represents the front
    face width (operationally horizontal edges), not the foreshortened bottom
    edge visible under perspective.

    Returns:
        (face_angle_deg, pair_mean_length) for diagnostics / weighting.
    """
    edges = _polygon_edges(polygon)
    pair_angles = [
        normalize_angle_deg((edges[0][0] + edges[2][0]) / 2.0),
        normalize_angle_deg((edges[1][0] + edges[3][0]) / 2.0),
    ]
    pair_lengths = [
        (edges[0][1] + edges[2][1]) / 2.0,
        (edges[1][1] + edges[3][1]) / 2.0,
    ]

    deviations = [_raw_angular_offset(angle, reference_angle) for angle in pair_angles]
    # Shelf-parallel pair is closest to reference (~0°), not the height axis (~90°).
    best_idx = 0 if deviations[0] <= deviations[1] else 1
    return pair_angles[best_idx], pair_lengths[best_idx]


def compute_operational_alignment(
    polygon: np.ndarray,
    reference_angle: float,
    threshold_deg: float = DEFAULT_ALIGNMENT_THRESHOLD_DEG,
) -> OperationalAlignment:
    """
    Measure operational misalignment of a detected OBB polygon.

    deviation_deg is the front-face rotation relative to the shelf/conveyor,
    NOT the raw YOLO xywhr angle or image-space cardinal offset alone.
    """
    face_angle, _ = shelf_parallel_face_angle(polygon, reference_angle)
    deviation = deviation_from_reference(face_angle, reference_angle)
    status = "ALIGNED" if deviation < threshold_deg else "MISALIGNED"
    return OperationalAlignment(
        face_angle_deg=face_angle,
        reference_angle_deg=reference_angle,
        deviation_deg=deviation,
        status=status,
    )


def extract_obb_angle(angle_rad: float) -> Tuple[float, float]:
    """
    Legacy helper: raw YOLO xywhr angle vs image cardinal axes.

    Prefer compute_operational_alignment() for robotics decisions. This
    function remains for debugging model rotation output only.
    """
    angle_deg = math.degrees(angle_rad)
    deviation_deg = min(angle_deg % 90.0, 90.0 - (angle_deg % 90.0))
    return angle_deg, deviation_deg


def snap_polygon_to_shelf_frame(
    polygon: np.ndarray,
    reference_angle: float,
) -> np.ndarray:
    """
    Rebuild a clean rotated rectangle aligned to the shelf reference.

    Useful when correcting annotations: preserves centroid and projected
    extents but removes operational rotation (perspective should not encode
    misalignment in the label).
    """
    polygon = polygon.astype(np.float32)
    cx = float(polygon[:, 0].mean())
    cy = float(polygon[:, 1].mean())
    rad = math.radians(reference_angle)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    proj_u: List[float] = []
    proj_v: List[float] = []
    for x, y in polygon:
        dx, dy = float(x) - cx, float(y) - cy
        proj_u.append(dx * cos_r + dy * sin_r)
        proj_v.append(-dx * sin_r + dy * cos_r)

    half_u = (max(proj_u) - min(proj_u)) / 2.0
    half_v = (max(proj_v) - min(proj_v)) / 2.0

    corners: List[List[float]] = []
    for su, sv in [(-half_u, -half_v), (half_u, -half_v), (half_u, half_v), (-half_u, half_v)]:
        x = cx + su * cos_r - sv * sin_r
        y = cy + su * sin_r + sv * cos_r
        corners.append([x, y])
    return np.array(corners, dtype=np.float32)
