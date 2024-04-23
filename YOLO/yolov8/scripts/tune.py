import wandb
from ultralytics import YOLO
# from wandb.integration.ultralytics import add_wandb_callback
from pathlib import Path

# Initialize wandb

base_dir = Path(__file__).resolve().parent.parent
# Path to the YOLOv8 model file
model_path = base_dir / 'util' / 'weights' / 'robocup1-15-tune13.pt'
# Load the model.
model = YOLO(model_path)

if __name__ == '__main__':
    # Ensure wandb is initialized with the proper project and entity
    wandb.init(project="Robocup24 Detection Training", entity="romelavision", resume="allow")
     
    try:
        # Use the model
        results = model.tune(
            project='Robocup24 Detection Training',
            data='robocup-tune.yaml',
            name='robocup1-15-tune13-tune',
            imgsz=640,
            epochs=30,
            batch=50,
            iterations=10,
            patience=5,
            cos_lr=True, 
            # lr0=0.001,
            optimizer='AdamW',
            close_mosaic=5,
            pretrained=True,
            freeze=14,
            save=True)

        # Assuming results contain the metrics 
        for epoch, metrics in enumerate(results):
            wandb.log(metrics)

    finally:
        # Finish the wandb run
        wandb.finish()
