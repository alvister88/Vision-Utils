import wandb
from ultralytics import YOLO
from pathlib import Path
import yaml

class ModelTuner:
    def __init__(self, model_weights_path, params_file_path):
        """
        Initializes the ModelTuner class with the paths to the model weights and parameter files.

        Args:
            model_weights_path (str): The path to the model weights file.
            params_file_path (str): The path to the hyperparameters YAML file.
        """
        self.model_path = model_weights_path
        self.params_path = params_file_path

    def load_hyperparameters(self):
        """
        Loads hyperparameters from a YAML file.

        Returns:
            dict: A dictionary containing the hyperparameters loaded from the YAML file.
        """
        with open(self.params_path, 'r') as file:
            return yaml.safe_load(file)

    def create_model(self):
        """
        Creates a YOLO model based on the specified model path.

        Returns:
            YOLO: A YOLO object representing the created model.
        """
        return YOLO(self.model_path)

    def configure_wandb(self, project_name, entity_name):
        """
        Initializes a wandb run with the specified project name and entity name.

        Args:
            project_name (str): The name of the project.
            entity_name (str): The name of the entity.
        """
        wandb.init(project=project_name, entity=entity_name, resume="allow")

    def tune_model(self, model, hyperparams):
        """
        Tunes a model using the provided hyperparameters.

        Args:
            model (YOLO): The model to be tuned.
            hyperparams (dict): A dictionary containing the hyperparameters for tuning.

        Returns:
            list: A list of dictionaries containing metrics for each epoch.
        """
        return model.tune(**hyperparams)

    def log_metrics_and_mosaic_to_wandb(self, model, results):
        """
        Logs metrics and a mosaic image to wandb for each epoch of the training process.

        Args:
            model (YOLO): The YOLO model used for validation inference.
            results (list): A list of dictionaries containing metrics for each epoch.
        """
        for epoch, metrics in enumerate(results):
            wandb.log(metrics)
            mosaic = model.val(verbose=False, imgsz=640)
            wandb_image = wandb.Image(mosaic, caption=f"Validation Mosaic - Epoch {epoch + 1}")
            wandb.log({"Validation Mosaic": wandb_image})

    def run(self, project_name, entity_name):
        """
        Executes the complete tuning process from initializing wandb to tuning the model and logging results.

        Args:
            project_name (str): The name of the project.
            entity_name (str): The name of the entity.
        """
        hyperparams = self.load_hyperparameters()
        model = self.create_model()
        self.configure_wandb(project_name, entity_name)

        try:
            results = self.tune_model(model, hyperparams)
            self.log_metrics_and_mosaic_to_wandb(results)
        finally:
            wandb.finish()

# Example of how to use the ModelTuner class
if __name__ == '__main__':
    # Set up paths
    base_dir = Path(__file__).resolve().parent
    model_weights_path = base_dir / 'weights' / 'yolov8m.pt'
    params_file_path = base_dir / 'config-examples' / 'tuning-example.yaml'

    # Initialize the ModelTuner
    tuner = ModelTuner(model_weights_path, params_file_path)

    # Run the tuning process
    tuner.run("Robocup24 Detection Tuning", "romelavision")
