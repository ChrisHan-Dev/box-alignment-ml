"""
Box alignment visualization pipeline for YOLOv8 OBB inference.

Operational objective
---------------------
Detect whether the carton's FRONT FACE is parallel to the shelf/conveyor
rails. This is a robotics placement task — not perspective-accurate geometry,
document detection, or photogrammetry.

Labels and post-inference metrics should encode *desired operational orientation*
(clean rotated rectangles aligned to shelf direction), not camera-space
trapezoids traced around perspective foreshortening.

Designed for batch image evaluation today and realtime camera / ROS2 integration
later. Core geometry lives in alignment_core.py.
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

from alignment_core import (
    AlignmentConfig,
    DEFAULT_ALIGNMENT_THRESHOLD_DEG,
    ReferenceMode,
    compute_operational_alignment,
    resolve_reference_angle,
)

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

# Default operational alignment config (shelf/conveyor reference frame)
DEFAULT_ALIGNMENT_CONFIG = AlignmentConfig(
    mode=ReferenceMode.AUTO_CONVEYOR,
    threshold_deg=DEFAULT_ALIGNMENT_THRESHOLD_DEG,
)

# Split-box merge (same physical box detected as left/right halves)
SPLIT_MERGE_MAX_Y_DIFF = 50.0
SPLIT_MERGE_MAX_ANGLE_DIFF = 15.0
SPLIT_MERGE_MAX_X_GAP = 280.0
MAX_BOXES_PER_FRAME = 1

# Visualization colors (BGR)
COLOR_ALIGNED = (0, 200, 0)
COLOR_MISALIGNED = (0, 0, 220)
COLOR_CENTER = (0, 255, 255)
COLOR_REFERENCE = (255, 180, 0)
COLOR_TEXT_BG = (30, 30, 30)


@dataclass(frozen=True)
class AlignmentResult:
    """Structured output for one OBB detection (ROS2-message friendly)."""

    polygon: np.ndarray  # shape (4, 2), float32 pixel coordinates
    center: Tuple[float, float]
    face_angle_deg: float  # shelf-parallel front-face axis
    reference_angle_deg: float  # shelf/conveyor direction used as horizontal
    deviation_deg: float  # operational deviation from shelf-parallel
    confidence: float
    class_id: int
    status: str  # "ALIGNED" | "MISALIGNED"

    # Backward-compatible aliases
    @property
    def angle_deg(self) -> float:
        return self.face_angle_deg


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


def _rebuild_detection(
    polygon: np.ndarray,
    confidence: float,
    class_id: int,
    image_bgr: Optional[np.ndarray] = None,
    config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG,
) -> AlignmentResult:
    """Build AlignmentResult using operational shelf-relative alignment."""
    polygon = polygon.astype(np.float32)
    center = (float(polygon[:, 0].mean()), float(polygon[:, 1].mean()))
    reference = resolve_reference_angle(image_bgr, config)
    metrics = compute_operational_alignment(
        polygon, reference, threshold_deg=config.threshold_deg
    )

    return AlignmentResult(
        polygon=polygon,
        center=center,
        face_angle_deg=metrics.face_angle_deg,
        reference_angle_deg=metrics.reference_angle_deg,
        deviation_deg=metrics.deviation_deg,
        confidence=confidence,
        class_id=class_id,
        status=metrics.status,
    )


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


def _merge_group(
    group: Sequence[AlignmentResult],
    image_bgr: Optional[np.ndarray] = None,
    config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG,
) -> AlignmentResult:
    """Merge split left/right detections into one min-area rectangle."""
    all_points = np.vstack([det.polygon for det in group]).astype(np.float32)
    merged_box = cv2.boxPoints(cv2.minAreaRect(all_points)).astype(np.float32)
    return _rebuild_detection(
        merged_box,
        confidence=max(det.confidence for det in group),
        class_id=group[0].class_id,
        image_bgr=image_bgr,
        config=config,
    )


def merge_split_detections(
    detections: Sequence[AlignmentResult],
    image_bgr: Optional[np.ndarray] = None,
    config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG,
) -> List[AlignmentResult]:
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
        ang_diff = abs(_normalize_angle_deg(det.face_angle_deg - prev.face_angle_deg))
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
        merged.append(
            _merge_group(group, image_bgr, config) if len(group) > 1 else group[0]
        )
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
    config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG,
) -> List[AlignmentResult]:
    """
    Parse Ultralytics OBB output into AlignmentResult objects.

    Alignment is computed from the detected polygon corners (obb.xyxyxyxy),
    measuring front-face rotation relative to the shelf/conveyor reference —
    not from raw xywhr angle alone.
    """
    if result.obb is None or len(result.obb) == 0:
        return []

    polygons = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
    detections: List[AlignmentResult] = []

    for i, box in enumerate(result.obb.data):
        _, _, _, _, _, conf, cls_id = box.tolist()
        polygon = polygons[i].astype(np.float32)
        detections.append(
            _rebuild_detection(
                polygon, float(conf), int(cls_id), image_bgr=None, config=config
            )
        )

    filtered = filter_detections(detections)
    merged = merge_split_detections(filtered, image_bgr=None, config=config)
    if len(merged) < len(filtered):
        print(f"  [DEBUG] split-merge: {len(filtered)} -> {len(merged)} detection(s)")

    single = enforce_single_box(merged)
    if len(single) < len(merged):
        print(f"  [DEBUG] single-box: {len(merged)} -> {len(single)} detection(s)")

    # Recompute with frame reference (conveyor Hough or fixed angle)
    final: List[AlignmentResult] = []
    for det in single:
        final.append(
            _rebuild_detection(
                det.polygon, det.confidence, det.class_id, image_bgr, config
            )
        )

    if len(final) < len(detections):
        print(f"  [DEBUG] post-filter: {len(detections)} -> {len(final)} detection(s)")

    for det in final:
        print(
            f"  [DEBUG] conf={det.confidence:.3f} | "
            f"ref={det.reference_angle_deg:.2f}° | "
            f"face={det.face_angle_deg:.2f}° | "
            f"shelf_dev={det.deviation_deg:.2f}° | "
            f"status={det.status}"
        )

    return final


def _draw_reference_line(
    canvas: np.ndarray,
    center: Tuple[float, float],
    reference_angle_deg: float,
    length: float = 120.0,
) -> None:
    """Draw shelf/conveyor reference direction through the box center."""
    cx, cy = center
    rad = math.radians(reference_angle_deg)
    dx = length * math.cos(rad)
    dy = length * math.sin(rad)
    p1 = (int(cx - dx), int(cy - dy))
    p2 = (int(cx + dx), int(cy + dy))
    cv2.line(canvas, p1, p2, COLOR_REFERENCE, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "SHELF",
        (p2[0] + 4, p2[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_REFERENCE,
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    image: np.ndarray,
    detections: Sequence[AlignmentResult],
    draw_reference: bool = True,
) -> np.ndarray:
    """
    Draw OBB polygons and operational alignment annotations.

    Overlay layout per detection (above the box center):
        DEV:   shelf-relative deviation (degrees) — actionable metric
        REF:   conveyor reference angle used for this frame
        CONF:  detection confidence
        STATUS: ALIGNED | MISALIGNED
    """
    canvas = image.copy()

    for det in detections:
        color = COLOR_ALIGNED if det.status == "ALIGNED" else COLOR_MISALIGNED
        pts = det.polygon.astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.circle(canvas, (cx, cy), radius=5, color=COLOR_CENTER, thickness=-1)

        if draw_reference:
            _draw_reference_line(canvas, det.center, det.reference_angle_deg)

        lines = [
            f"DEV: {det.deviation_deg:.1f}",
            f"REF: {det.reference_angle_deg:.1f}",
            f"CONF: {det.confidence:.2f}",
            f"STATUS: {det.status}",
        ]
        _draw_label_block(canvas, lines, anchor=(cx, cy - 80), accent=color)

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
    config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG,
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

    detections = parse_obb_results(results[0], image_bgr=frame, config=config)
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
    config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG,
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
    print(f"[INFO] Reference mode: {config.mode.value}")
    print(f"[INFO] Alignment threshold: {config.threshold_deg}° (shelf-relative)")

    print(f"[INFO] Device: {resolve_device(device)}")

    for image_path in collect_images(source_path):
        result_path = visualize_image(
            model, image_path, out_path, conf=conf, device=device, config=config
        )
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
    config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG,
) -> Tuple[np.ndarray, List[AlignmentResult]]:
    """
    Single-frame pipeline for live camera input.

    Usage in a future ROS2 node:
        annotated, detections = process_frame(model, cv_bridge_image)
        for det in detections:
            publish_alignment_status(det.status, det.deviation_deg)
    """
    results = run_inference(model, frame_bgr, conf=conf, device=device)
    detections = (
        parse_obb_results(results[0], image_bgr=frame_bgr, config=config) if results else []
    )
    annotated = draw_overlay(frame_bgr, detections)
    return annotated, detections


if __name__ == "__main__":
    visualize_batch()
