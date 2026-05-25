"""
Random-sample inference review: compare predictions vs ground-truth labels.

Usage:
    python3 sample_review.py                  # 20 random images
    python3 sample_review.py --count 30
    python3 sample_review.py --include test/images/e57701fc*.jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from alignment_core import AlignmentConfig, compute_operational_alignment, resolve_reference_angle
from visualize_alignment import (
    DEFAULT_OUTPUT_DIR,
    draw_overlay,
    load_model,
    parse_obb_results,
    resolve_model_path,
)

DATASET_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = DATASET_ROOT / "runs/obb/alignment_sample_review"


def _load_gt_metrics(
    lbl_path: Path, img_shape: Tuple[int, int, int], ref: float, threshold: float
) -> Optional[Tuple[float, str]]:
    if not lbl_path.is_file():
        return None
    parts = lbl_path.read_text().strip().split()
    if len(parts) < 9:
        return None
    h, w = img_shape[:2]
    coords = list(map(float, parts[1:]))
    pts = np.array([[coords[i] * w, coords[i + 1] * h] for i in range(0, 8, 2)], dtype=np.float32)
    m = compute_operational_alignment(pts, ref, threshold)
    return m.deviation_deg, m.status


def _collect_images() -> List[Path]:
    images: List[Path] = []
    for split in ("train", "valid", "test"):
        img_dir = DATASET_ROOT / split / "images"
        if img_dir.is_dir():
            images.extend(sorted(img_dir.glob("*.jpg")))
    return images


def _pred_gt_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    _, inter = cv2.intersectConvexConvex(pred.astype(np.float32), gt.astype(np.float32))
    if inter is None or len(inter) < 3:
        return 0.0
    inter_a = cv2.contourArea(inter.astype(np.float32))
    union = cv2.contourArea(pred.astype(np.float32)) + cv2.contourArea(gt.astype(np.float32)) - inter_a
    return inter_a / union if union > 0 else 0.0


def run_review(
    count: int = 20,
    seed: int = 42,
    conf: float = 0.40,
    include: Optional[Sequence[str]] = None,
    output_dir: Path = DEFAULT_OUT,
) -> None:
    cfg = AlignmentConfig()
    model = load_model()
    print(f"[INFO] Weights: {resolve_model_path()}")
    print(f"[INFO] Output : {output_dir}\n")

    all_images = _collect_images()
    must: List[Path] = []
    if include:
        for pattern in include:
            must.extend(DATASET_ROOT.glob(pattern))
    must = [p.resolve() for p in must if p.is_file()]

    pool = [p for p in all_images if p not in must]
    random.seed(seed)
    sample = must + random.sample(pool, min(max(count - len(must), 0), len(pool)))

    output_dir.mkdir(parents=True, exist_ok=True)
    header = f"{'IMAGE':<42} {'SPL':<5} {'CONF':>5} {'IoU':>5} {'P_DEV':>6} {'G_DEV':>6} {'PRED':>10} {'GT':>10} {'OK':>4}"
    print(header)
    print("-" * len(header))

    stats = {"det": 0, "match": 0, "low_iou": 0, "pred_mis": 0}

    for img_path in sample:
        split = img_path.parent.parent.name
        lbl_path = DATASET_ROOT / split / "labels" / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        ref = resolve_reference_angle(img, cfg)
        gt = _load_gt_metrics(lbl_path, img.shape, ref, cfg.threshold_deg)
        gt_dev, gt_status = gt if gt else (None, None)

        r = model.predict(str(img_path), conf=conf, verbose=False)[0]
        dets = parse_obb_results(r, image_bgr=img)

        if dets:
            d = dets[0]
            stats["det"] += 1
            if d.status == "MISALIGNED":
                stats["pred_mis"] += 1

            iou = 0.0
            if lbl_path.is_file():
                parts = lbl_path.read_text().strip().split()
                if len(parts) >= 9:
                    h, w = img.shape[:2]
                    coords = list(map(float, parts[1:]))
                    gt_pts = np.array(
                        [[coords[i] * w, coords[i + 1] * h] for i in range(0, 8, 2)],
                        dtype=np.float32,
                    )
                    iou = _pred_gt_iou(d.polygon, gt_pts)
                    if iou < 0.3:
                        stats["low_iou"] += 1

            agree = d.status == gt_status if gt_status else None
            if agree:
                stats["match"] += 1

            overlay = draw_overlay(img, dets)
            cv2.imwrite(str(output_dir / img_path.name), overlay)
            ok = "YES" if agree else ("NO" if agree is False else "N/A")
            print(
                f"{img_path.name[:42]:<42} {split:<5} {d.confidence:5.2f} {iou:5.2f} "
                f"{d.deviation_deg:6.2f} {gt_dev or -1:6.2f} {d.status:>10} "
                f"{gt_status or 'NONE':>10} {ok:>4}"
            )
        else:
            cv2.imwrite(str(output_dir / img_path.name), img)
            print(
                f"{img_path.name[:42]:<42} {split:<5} {'---':>5} {'---':>5} "
                f"{'---':>6} {gt_dev or -1:6.2f} {'NO_DET':>10} {gt_status or 'NONE':>10} {'---':>4}"
            )

    print(
        f"\n[INFO] {stats['det']} detected | {stats['match']} status match GT | "
        f"{stats['low_iou']} IoU<0.3 | {stats['pred_mis']} pred MISALIGNED"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Random sample alignment review")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_review(
        count=args.count,
        seed=args.seed,
        conf=args.conf,
        include=args.include or None,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
