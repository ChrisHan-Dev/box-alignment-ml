from pathlib import Path

from ultralytics import YOLO

DATASET_ROOT = Path(__file__).resolve().parent
weights = sorted(
    (DATASET_ROOT / "runs/obb").glob("**/weights/best.pt"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)[0]

model = YOLO(str(weights))
print(f"Using weights: {weights}")

results = model.predict(source="test/images", save=True, conf=0.40)
print(f"Inference complete — {len(results)} image(s)")
