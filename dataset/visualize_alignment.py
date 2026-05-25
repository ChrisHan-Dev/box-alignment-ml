"""
Box alignment visualization pipeline for YOLOv8 OBB inference.

Designed for batch image evaluation today and realtime camera / ROS2 integration
later. Core logic is split into small functions that can be wrapped by a ROS2
node without rewriting the geometry or drawing code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ---------------------------------------------------------------------------
# Paths (relative to this script / dataset root)
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(__file__).resolve().parent
RUNS_OBB_DIR = DATASET_ROOT / "runs/obb"
DEFAULT_SOURCE = DATASET_ROOT / "test/images"
DEFAULT_OUTPUT_DIR = RUNS_OBB_DIR / "alignment_predict"

# Inference / post-processing defaults
DEFAULT_CONF = 0.40
NMS_IOU_THRESHOLD = 0.30
MIN_AREA_RATIO = 0.40  # drop detections smaller than 40% of largest in frame

# Alignment threshold: box bottom edge vs conveyor rail direction (degrees)
ALIGNMENT_THRESHOLD_DEG = 6.0

# Split-box merge (same physical box detected as left/right halves)
SPLIT_MERGE_MAX_Y_DIFF = 50.0
SPLIT_MERGE_MAX_ANGLE_DIFF = 15.0
SPLIT_MERGE_MAX_X_GAP = 280.0
MAX_BOXES_PER_FRAME = 1

# Visualization colors (BGR)
COLOR_ALIGNED = (0, 200, 0)
COLOR_MISALIGNED = (0, 0, 220)
COLOR_CENTER = (0, 255, 255)
COLOR_TEXT_BG = (30, 30, 30)


@dataclass(frozen=True)
class AlignmentResult:
    """Structured output for one OBB detection (ROS2-message friendly)."""

    polygon: np.ndarray  # shape (4, 2), float32 pixel coordinates
    center: Tuple[float, float]
    angle_deg: float
    deviation_deg: float
    confidence: float
    class_id: int
    status: str  # "ALIGNED" | "MISALIGNED"


def resolve_device(device: Optional[Union[int, str]] = None) -> Union[int, str]:
    """Pick CUDA when available; fall back to CPU (e.g. CI / headless WSL)."""
    if device is not None:
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def resolve_model_path(model_path: Optional[Union[str, Path]] = None) -> Path:
    """Use explicit path or the most recently trained runs/obb/train-*/weights/best.pt."""
    if model_path is not None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model weights not found: {path}")
        return path

    candidates = sorted(
        RUNS_OBB_DIR.glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    fallback = RUNS_OBB_DIR / "train-3/weights/best.pt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No trained weights found under {RUNS_OBB_DIR}")


def load_model(model_path: Optional[Union[str, Path]] = None) -> YOLO:
    """Load a trained YOLOv8 OBB checkpoint."""
    path = resolve_model_path(model_path)
    print(f"[INFO] Loading weights: {path}")
    return YOLO(str(path))


def run_inference(
    model: YOLO,
    source: Union[str, Path],
    conf: float = 0.25,
    device: Optional[Union[int, str]] = None,
) -> List[Results]:
    """
    Run OBB inference on an image path, directory, or numpy frame.

    Returns a list of Ultralytics Results objects (one per input image).
    For realtime use, pass a single BGR numpy array as `source`.
    """
    return model.predict(
        source=str(source),
        conf=conf,
        device=resolve_device(device),
        verbose=False,
    )


def _normalize_angle_deg(deg: float) -> float:
    """Map any angle to [-90, 90) for line-direction comparison."""
    while deg >= 90.0:
        deg -= 180.0
    while deg < -90.0:
        deg += 180.0
    return deg


def estimate_conveyor_angle(image_bgr: np.ndarray) -> float:
    """
    Estimate conveyor rail / roller direction from the lower belt ROI.

    Uses weighted Hough lines in the bottom ~45% of the frame. The returned
    angle is the dominant near-horizontal direction of the belt in image space.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    roi = cv2.GaussianBlur(gray[int(height * 0.55) :, :], (5, 5), 0)
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
        angle = _normalize_angle_deg(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if abs(angle) < 35.0 and length > width // 8:
            rad = math.radians(angle)
            sum_x += length * math.cos(rad)
            sum_y += length * math.sin(rad)
            total_weight += length

    if total_weight == 0.0:
        return 0.0
    return _normalize_angle_deg(math.degrees(math.atan2(sum_y, sum_x)))


def _polygon_edges(polygon: np.ndarray) -> List[Tuple[float, float, float]]:
    """Return (angle_deg, mid_y, length) for each polygon edge."""
    edges: List[Tuple[float, float, float]] = []
    for idx in range(4):
        p1 = polygon[idx]
        p2 = polygon[(idx + 1) % 4]
        mid_y = float((p1[1] + p2[1]) / 2.0)
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        length = math.hypot(dx, dy)
        angle = _normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
        edges.append((angle, mid_y, length))
    return edges


def bottom_edge_angle(polygon: np.ndarray) -> float:
    """Angle of the lowest edge of the OBB (closest to the conveyor surface)."""
    return max(_polygon_edges(polygon), key=lambda edge: edge[1])[0]


def deviation_from_reference(line_angle: float, reference_angle: float) -> float:
    """Smallest angular difference between two undirected line directions."""
    diff = abs(_normalize_angle_deg(line_angle - reference_angle))
    return min(diff, abs(90.0 - diff))


def extract_obb_angle(angle_rad: float) -> Tuple[float, float]:
    """
    Convert YOLO OBB rotation to degrees and horizontal-alignment deviation.

    YOLO OBB xywhr angle (radians):
        - Stored in obb.data[:, 4] and obb.xywhr[:, 4].
        - Rotation of the box width axis relative to the image x-axis.
        - Range is typically (-pi/2, pi/2] in Ultralytics; always convert with
          math.degrees() before comparing to degree thresholds.

    Deviation from horizontal alignment:
        A front-view box that is perfectly axis-aligned with the conveyor has
        rotation 0°, 90°, 180°, ... We measure the smallest offset to the
        nearest cardinal direction (0° or 90°), which is invariant to the
        width/height swap ambiguity in OBB representations.

        deviation = min(angle_deg % 90, 90 - (angle_deg % 90))

    Returns:
        (angle_deg, deviation_deg)
    """
    angle_deg = math.degrees(angle_rad)
    deviation_deg = min(angle_deg % 90.0, 90.0 - (angle_deg % 90.0))
    return angle_deg, deviation_deg


def compute_alignment(
    deviation_deg: float,
    threshold_deg: float = ALIGNMENT_THRESHOLD_DEG,
) -> str:
    """
    Classify box alignment from angular deviation.

    Rules:
        deviation < threshold  -> ALIGNED
        deviation >= threshold -> MISALIGNED
    """
    if deviation_deg < threshold_deg:
        return "ALIGNED"
    return "MISALIGNED"


def _polygon_area(polygon: np.ndarray) -> float:
    return float(cv2.contourArea(polygon.astype(np.float32)))


def _polygon_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """Intersection-over-union for two convex OBB polygons."""
    _, intersect = cv2.intersectConvexConvex(p1.astype(np.float32), p2.astype(np.float32))
    if intersect is None or len(intersect) < 3:
        return 0.0
    inter = cv2.contourArea(intersect.astype(np.float32))
    a1 = _polygon_area(p1)
    a2 = _polygon_area(p2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _rebuild_detection(
    polygon: np.ndarray,
    confidence: float,
    class_id: int,
    image_bgr: Optional[np.ndarray] = None,
) -> AlignmentResult:
    """Build AlignmentResult from a polygon, optionally using conveyor-relative angle."""
    polygon = polygon.astype(np.float32)
    center = (float(polygon[:, 0].mean()), float(polygon[:, 1].mean()))

    if image_bgr is not None:
        conveyor_angle = estimate_conveyor_angle(image_bgr)
        box_edge = bottom_edge_angle(polygon)
        deviation_deg = deviation_from_reference(box_edge, conveyor_angle)
        angle_deg = box_edge
    else:
        rect = cv2.minAreaRect(polygon)
        angle_deg = _normalize_angle_deg(float(rect[2]))
        deviation_deg = min(abs(angle_deg) % 90.0, 90.0 - (abs(angle_deg) % 90.0))

    status = compute_alignment(deviation_deg)
    return AlignmentResult(
        polygon=polygon,
        center=center,
        angle_deg=angle_deg,
        deviation_deg=deviation_deg,
        confidence=confidence,
        class_id=class_id,
        status=status,
    )


def _merge_group(group: Sequence[AlignmentResult]) -> AlignmentResult:
    """Merge split left/right detections into one min-area rectangle."""
    all_points = np.vstack([det.polygon for det in group]).astype(np.float32)
    merged_box = cv2.boxPoints(cv2.minAreaRect(all_points)).astype(np.float32)
    return _rebuild_detection(
        merged_box,
        confidence=max(det.confidence for det in group),
        class_id=group[0].class_id,
    )


def merge_split_detections(detections: Sequence[AlignmentResult]) -> List[AlignmentResult]:
    """
    Merge side-by-side split detections (low IoU, same row) into one OBB.

    Split boxes often have IoU ~0.02 because the model predicts separate left
    and right halves with a gap between them — standard NMS cannot merge them.
    """
    if len(detections) <= 1:
        return list(detections)

    ordered = sorted(detections, key=lambda det: det.center[0])
    groups: List[List[AlignmentResult]] = [[ordered[0]]]

    for det in ordered[1:]:
        prev = groups[-1][-1]
        cy_diff = abs(det.center[1] - prev.center[1])
        ang_diff = abs(_normalize_angle_deg(det.angle_deg - prev.angle_deg))
        x_gap = det.center[0] - prev.center[0]
        if (
            cy_diff <= SPLIT_MERGE_MAX_Y_DIFF
            and ang_diff <= SPLIT_MERGE_MAX_ANGLE_DIFF
            and x_gap <= SPLIT_MERGE_MAX_X_GAP
        ):
            groups[-1].append(det)
        else:
            groups.append([det])

    merged: List[AlignmentResult] = []
    for group in groups:
        merged.append(_merge_group(group) if len(group) > 1 else group[0])
    return merged


def enforce_single_box(detections: Sequence[AlignmentResult]) -> List[AlignmentResult]:
    """Keep at most one box — warehouse belt carries one target box per frame."""
    if len(detections) <= MAX_BOXES_PER_FRAME:
        return list(detections)
    best = max(detections, key=lambda det: _polygon_area(det.polygon))
    return [best]


def filter_detections(
    detections: Sequence[AlignmentResult],
    nms_iou: float = NMS_IOU_THRESHOLD,
    min_area_ratio: float = MIN_AREA_RATIO,
) -> List[AlignmentResult]:
    """
    Post-process raw detections: NMS for split boxes, drop tiny false positives.

    - Greedy NMS by confidence removes overlapping duplicate boxes (split detection).
    - Minimum area ratio removes thin vertical strips vs the primary detection.
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[AlignmentResult] = []

    for det in ordered:
        if any(_polygon_iou(det.polygon, k.polygon) > nms_iou for k in kept):
            continue
        kept.append(det)

    if not kept:
        return []

    max_area = max(_polygon_area(d.polygon) for d in kept)
    return [d for d in kept if _polygon_area(d.polygon) >= max_area * min_area_ratio]


def parse_obb_results(
    result: Results,
    image_bgr: Optional[np.ndarray] = None,
) -> List[AlignmentResult]:
    """
    Parse Ultralytics OBB output into AlignmentResult objects.

    YOLO OBB output formats (Ultralytics Results.obb):
        obb.data       : (N, 7) tensor
                         [x_center, y_center, w, h, angle_rad, conf, class_id]
                         All spatial values are in pixel coordinates for the
                         original (letterbox-resized) image.

        obb.xywhr      : (N, 5) — center, size, angle only (same angle_rad).

        obb.xyxyxyxy   : (N, 4, 2) — four polygon corner points in pixel space.
                         Corners follow the rotated rectangle; use these for
                         drawing because they match the visible box exactly.

        obb.conf       : (N,) detection confidence in [0, 1].
        obb.cls        : (N,) class index.

    Angle interpretation:
        angle_rad is rotation in radians, NOT degrees. Positive rotation follows
        the standard image coordinate system (y increases downward).
    """
    if result.obb is None or len(result.obb) == 0:
        return []

    polygons = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
    detections: List[AlignmentResult] = []

    for i, box in enumerate(result.obb.data):
        # box layout: [cx, cy, w, h, angle_rad, conf, cls]
        _, _, _, _, angle_rad, conf, cls_id = box.tolist()
        polygon = polygons[i].astype(np.float32)
        detections.append(
            _rebuild_detection(polygon, float(conf), int(cls_id), image_bgr=None)
        )

    filtered = filter_detections(detections)
    merged = merge_split_detections(filtered)
    if len(merged) < len(filtered):
        print(f"  [DEBUG] split-merge: {len(filtered)} -> {len(merged)} detection(s)")

    single = enforce_single_box(merged)
    if len(single) < len(merged):
        print(f"  [DEBUG] single-box: {len(merged)} -> {len(single)} detection(s)")

    # Recompute conveyor-relative alignment on final polygon(s)
    final: List[AlignmentResult] = []
    for det in single:
        if image_bgr is not None:
            final.append(
                _rebuild_detection(det.polygon, det.confidence, det.class_id, image_bgr)
            )
        else:
            final.append(det)

    if len(final) < len(detections):
        print(f"  [DEBUG] post-filter: {len(detections)} -> {len(final)} detection(s)")

    for det in final:
        print(
            f"  [DEBUG] conf={det.confidence:.3f} | "
            f"box_edge={det.angle_deg:.2f}° | "
            f"deviation={det.deviation_deg:.2f}° | "
            f"status={det.status}"
        )

    return final


def draw_overlay(
    image: np.ndarray,
    detections: Sequence[AlignmentResult],
) -> np.ndarray:
    """
    Draw OBB polygons and annotation text on a copy of the input frame.

    Overlay layout per detection (above the box center):
        ANGLE: <deviation>°   (uses deviation — the actionable alignment metric)
        CONF:  <confidence>
        STATUS: ALIGNED | MISALIGNED
    """
    canvas = image.copy()

    for det in detections:
        color = COLOR_ALIGNED if det.status == "ALIGNED" else COLOR_MISALIGNED
        pts = det.polygon.astype(np.int32).reshape((-1, 1, 2))

        # Rotated OBB polygon (4 corners from obb.xyxyxyxy)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        # Center point (centroid of polygon corners)
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.circle(canvas, (cx, cy), radius=5, color=COLOR_CENTER, thickness=-1)

        lines = [
            f"ANGLE: {det.deviation_deg:.1f}",
            f"CONF: {det.confidence:.2f}",
            f"STATUS: {det.status}",
        ]  # ANGLE = deviation from horizontal/cardinal alignment (degrees)
        _draw_label_block(canvas, lines, anchor=(cx, cy - 60), accent=color)

    return canvas


def _draw_label_block(
    image: np.ndarray,
    lines: Sequence[str],
    anchor: Tuple[int, int],
    accent: Tuple[int, int, int],
) -> None:
    """Render a multi-line text block with a dark background."""
    x, y = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 22
    padding = 6

    widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
    block_w = max(widths) + padding * 2
    block_h = line_height * len(lines) + padding * 2

    x1 = max(x - block_w // 2, 0)
    y1 = max(y - block_h, 0)
    x2 = min(x1 + block_w, image.shape[1] - 1)
    y2 = min(y1 + block_h, image.shape[0] - 1)

    cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_TEXT_BG, thickness=-1)
    cv2.rectangle(image, (x1, y1), (x2, y2), accent, thickness=1)

    for idx, line in enumerate(lines):
        ty = y1 + padding + (idx + 1) * line_height - 6
        cv2.putText(image, line, (x1 + padding, ty), font, font_scale, (255, 255, 255), thickness)


def visualize_image(
    model: YOLO,
    image_path: Path,
    output_dir: Path,
    conf: float = DEFAULT_CONF,
    device: Optional[Union[int, str]] = None,
) -> Optional[Path]:
    """Run inference on one image, draw overlays, and save the result."""
    print(f"\n[INFO] Processing: {image_path.name}")

    results = run_inference(model, image_path, conf=conf, device=device)
    if not results:
        print("[WARN] No results returned.")
        return None

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None

    detections = parse_obb_results(results[0], image_bgr=frame)
    if not detections:
        print("[WARN] No OBB detections — saving original frame with warning tag.")
        cv2.putText(
            frame,
            "NO DETECTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
    else:
        frame = draw_overlay(frame, detections)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / image_path.name
    cv2.imwrite(str(out_path), frame)
    print(f"[INFO] Saved: {out_path}")
    return out_path


def collect_images(source: Path) -> List[Path]:
    """Collect image files from a file or directory."""
    if source.is_file():
        return [source]
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(p for p in source.iterdir() if p.suffix.lower() in extensions)


def visualize_batch(
    model_path: Optional[Union[str, Path]] = None,
    source: Union[str, Path] = DEFAULT_SOURCE,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    conf: float = DEFAULT_CONF,
    device: Optional[Union[int, str]] = None,
) -> List[Path]:
    """Process all images under `source` and write overlays to `output_dir`."""
    weights = resolve_model_path(model_path)
    model = load_model(weights)
    source_path = Path(source)
    out_path = Path(output_dir)
    saved: List[Path] = []

    print(f"[INFO] Model : {weights}")
    print(f"[INFO] Source: {source_path}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Alignment threshold: {ALIGNMENT_THRESHOLD_DEG}°")

    print(f"[INFO] Device: {resolve_device(device)}")

    for image_path in collect_images(source_path):
        result_path = visualize_image(model, image_path, out_path, conf=conf, device=device)
        if result_path is not None:
            saved.append(result_path)

    print(f"\n[INFO] Done — {len(saved)} image(s) saved to {out_path}")
    return saved


# ---------------------------------------------------------------------------
# Realtime hook (future ROS2 / Gazebo camera node)
# ---------------------------------------------------------------------------
def process_frame(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float = DEFAULT_CONF,
    device: Optional[Union[int, str]] = None,
) -> Tuple[np.ndarray, List[AlignmentResult]]:
    """
    Single-frame pipeline for live camera input.

    Usage in a future ROS2 node:
        annotated, detections = process_frame(model, cv_bridge_image)
        for det in detections:
            publish_alignment_status(det.status, det.deviation_deg)
    """
    results = run_inference(model, frame_bgr, conf=conf, device=device)
    detections = parse_obb_results(results[0], image_bgr=frame_bgr) if results else []
    annotated = draw_overlay(frame_bgr, detections)
    return annotated, detections


if __name__ == "__main__":
    visualize_batch()
