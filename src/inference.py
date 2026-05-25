"""
Inference entry point — train-4 OBB weights + visualize_alignment post-processing.

See dataset/alignment_core.py for geometry and dataset/visualize_alignment.py
for filtering, split-merge, and shelf-relative status.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import cv2

_DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

from alignment_core import AlignmentConfig, ReferenceMode  # noqa: E402
from visualize_alignment import (  # noqa: E402
    AlignmentResult,
    DEFAULT_ALIGNMENT_CONFIG,
    DEFAULT_CONF,
    load_model,
    parse_obb_results,
    resolve_device,
    run_inference,
)

DEFAULT_WEIGHTS = _DATASET_ROOT / "runs/obb/train-4/weights/best.pt"

DEFAULT_CONFIG = AlignmentConfig(
    mode=ReferenceMode.AUTO_CONVEYOR,
    threshold_deg=DEFAULT_ALIGNMENT_CONFIG.threshold_deg,
)

_model = None


def _get_model():
    global _model
    if _model is None:
        if not DEFAULT_WEIGHTS.is_file():
            raise FileNotFoundError(f"Trained weights not found: {DEFAULT_WEIGHTS}")
        _model = load_model(DEFAULT_WEIGHTS)
    return _model


def detect_alignments(
    image_path: str,
    conf: float = DEFAULT_CONF,
    config: AlignmentConfig | None = None,
) -> List[AlignmentResult]:
    """
    Run OBB inference with the same post-processing as visualize_alignment.py.

    Returns parsed detections (split-merge, single-box filter, shelf-relative status).
    """
    cfg = config or DEFAULT_CONFIG
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    model = _get_model()
    results = run_inference(model, image_path, conf=conf, device=resolve_device())
    if not results:
        return []
    return parse_obb_results(results[0], image_bgr=frame, config=cfg)


def analyze_box_alignment(
    image_path: str,
    max_allow_deviation: float = 6.0,
    config: AlignmentConfig | None = None,
    conf: float = DEFAULT_CONF,
) -> bool:
    """
    Detect cartons and evaluate operational alignment with the shelf/conveyor.

    Returns True when teleoperator intervention is required (MISALIGNED box).
    """
    cfg = config or AlignmentConfig(
        mode=DEFAULT_CONFIG.mode,
        threshold_deg=max_allow_deviation,
    )
    print(f"[INFO] Chạy AI phân tích ảnh: {image_path}")
    print(f"[INFO] Weights: {DEFAULT_WEIGHTS}")

    try:
        detections = detect_alignments(image_path, conf=conf, config=cfg)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return False

    flag_teleoperator = False

    if detections:
        for det in detections:
            if det.status == "MISALIGNED":
                status = f"BỊ NGHIÊNG/LỆCH CHÉO (shelf_dev={det.deviation_deg:.2f}°)"
                flag_teleoperator = True
            else:
                status = f"THẲNG HÀNG CHUẨN (shelf_dev={det.deviation_deg:.2f}°)"

            print(
                f" -> Phát hiện vật thể (Độ tự tin: {det.confidence * 100:.1f}%) | "
                f"ref={det.reference_angle_deg:.1f}° | face={det.face_angle_deg:.1f}° | "
                f"{status}"
            )
    else:
        print("[WARN] Không tìm thấy hộp hoặc vật thể nào trong tầm nhìn.")

    if flag_teleoperator:
        trigger_teleoperation_route()

    return flag_teleoperator


def trigger_teleoperation_route() -> None:
    print("\n[ALERT] !!! CẢNH BÁO: HỘP TRÊN BĂNG CHUYỀN BỊ NGHIÊNG/LỆCH QUÁ MỨC !!!")
    print("[ALERT] Đang gửi tín hiệu yêu cầu Teleoperator thủ công can thiệp...\n")


if __name__ == "__main__":
    analyze_box_alignment("src/images/test_box.webp")
