import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
from pathlib import Path
import yaml

def load_hyperparameters(yaml_file):
    """
    Loads hyperparameters from a YAML file.

    Args:
        yaml_file (str): The path to the YAML file containing the hyperparameters.

    Returns:
        dict: A dictionary containing the hyperparameters loaded from the YAML file.
    """
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def create_model(model_path):
    """
    Creates a model based on the specified model path.

    Args:
        model_path (str): The path to the model.

    Returns:
        YOLO: A YOLO object representing the created model.
    """
    return YOLO(model_path)

def configure_wandb(project_name, entity_name):
    """
    Initializes a wandb run with the specified project name and entity name.

    Args:
        project_name (str): The name of the project.
        entity_name (str): The name of the entity.

    Returns:
        None
    """
    wandb.init(project=project_name, entity=entity_name, resume="allow")

def train_model(model, hyperparams):
    """
    Trains a model using the provided hyperparameters.

    Args:
        model (object): The model to be trained.
        hyperparams (dict): A dictionary containing the hyperparameters for training.

    Returns:
        object: The trained model.
    """
    return model.train(**hyperparams)

def log_metrics_and_mosaic_to_wandb(model, results):
    """
    Logs metrics and a mosaic image to wandb for each epoch of the training process.

    Args:
        model (YOLO): The YOLO model used for validation inference.
        results (list): A list of dictionaries containing metrics for each epoch.

    Returns:
        None
    """
    for epoch, metrics in enumerate(results):
        # Log the metrics for the epoch
        wandb.log(metrics)

        # Generate the mosaic of the validation inference
        mosaic = model.val(verbose=False, imgsz=640)

        # Log the mosaic image to wandb
        wandb_image = wandb.Image(mosaic, caption=f"Validation Mosaic - Epoch {epoch + 1}")
        wandb.log({"Validation Mosaic": wandb_image})

def main():
    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / 'util' / 'weights' / 'yolov8m.pt'
    params_path = base_dir / 'util' / 'args.yaml'

    # Load hyperparameters from YAML file
    hyperparams = load_hyperparameters(params_path)

    # Create the YOLOv8 model
    model = create_model(model_path)

    # Configure wandb
    configure_wandb("Robocup24 Detection Training", "romelavision")

    try:
        # Train the model with hyperparameters
        results = train_model(model, hyperparams)

        # Log metrics and mosaic to wandb
        log_metrics_and_mosaic_to_wandb(model, results)
    finally:
        # Finish the wandb run
        wandb.finish()

if __name__ == '__main__':
    main()