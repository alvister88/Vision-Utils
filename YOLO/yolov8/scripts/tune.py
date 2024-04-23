import wandb
from ultralytics import YOLO
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

def tune_model(model, hyperparams):
    """
    Tunes a model using the provided hyperparameters.

    Args:
        model (object): The model to be tuned.
        hyperparams (dict): A dictionary containing the hyperparameters for tuning.

    Returns:
        list: A list of dictionaries containing metrics for each epoch.
    """
    return model.tune(**hyperparams)

def log_metrics_to_wandb(results):
    """
    Logs metrics to wandb for each epoch of the tuning process.

    Args:
        results (list): A list of dictionaries containing metrics for each epoch.

    Returns:
        None
    """
    for epoch, metrics in enumerate(results):
        wandb.log(metrics)

def main():
    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / ''
    params_path = base_dir / ''

    # Load hyperparameters from YAML file
    hyperparams = load_hyperparameters(params_path)

    # Create the YOLO model
    model = create_model(model_path)

    # Configure wandb
    configure_wandb("Robocup24 Detection Tuning", "romelavision")

    try:
        # Tune the model with hyperparameters
        results = tune_model(model, hyperparams)

        # Log metrics to wandb
        log_metrics_to_wandb(results)
    finally:
        # Finish the wandb run
        wandb.finish()

if __name__ == '__main__':
    main()
