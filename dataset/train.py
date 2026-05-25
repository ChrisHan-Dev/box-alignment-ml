from pathlib import Path

from ultralytics import YOLO

DATASET_ROOT = Path(__file__).resolve().parent

model = YOLO("yolov8n-obb.pt")

model.train(
    data=str(DATASET_ROOT / "data.yaml"),
    epochs=60,
    imgsz=640,
    batch=4,
    device=0,
    workers=2,
    project=str(DATASET_ROOT / "runs" / "obb"),
    name="train",
    patience=15,
    cache=False,
    exist_ok=False,
)
