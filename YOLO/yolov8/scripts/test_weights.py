import wandb
from ultralytics import YOLO
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
# Path to the YOLOv8 model file
model_path = base_dir / 'util' / 'weights' / 'yolov8m.pt'
weight_name = 'robocup3-1.pt'
custom_path = base_dir / 'util' / 'weights' / f'{weight_name}'
# Load the model.
model = YOLO(custom_path)


if __name__ == '__main__':


    # Use the model
    results = model.val(
        data='robocup-test.yaml',
        imgsz=640,
        name=f'{weight_name}-test')

   