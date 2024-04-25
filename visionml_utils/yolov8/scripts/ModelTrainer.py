import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
from pathlib import Path
import yaml

class ModelTrainer:
    def __init__(self, model_weights_path, hyperparams_file_path):
        """
        Initializes the YOLOTrainer class with the directory paths and filenames.

        Args:
            model_weights_path (str): The path to the model weights file.
            params_file_path (str): The path to the hyperparameters YAML file.
        """
        self.model_path = model_weights_path
        self.hyperparams_path = hyperparams_file_path

    def load_hyperparameters(self):
        """
        Loads hyperparameters from a YAML file.

        Returns:
            dict: A dictionary containing the hyperparameters loaded from the YAML file.
        """
        with open(self.hyperparams_path, 'r') as file:
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

    def train_model(self, model, hyperparams):
        """
        Trains a model using the provided hyperparameters.

        Args:
            model (YOLO): The model to be trained.
            hyperparams (dict): A dictionary containing the hyperparameters for training.

        Returns:
            list: The results of the training.
        """
        return model.train(**hyperparams)

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
        Executes the complete training process from initializing wandb to training and logging.

        Args:
            project_name (str): The name of the project.
            entity_name (str): The name of the entity.
        """
        hyperparams = self.load_hyperparameters()
        model = self.create_model()
        self.configure_wandb(project_name, entity_name)

        try:
            results = self.train_model(model, hyperparams)
            self.log_metrics_and_mosaic_to_wandb(model, results)
        finally:
            wandb.finish()

# Example of how to use the YOLOTrainer class
if __name__ == '__main__':
    # Set up paths
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / 'weights' / 'yolov8m.pt'
    hyperparams_path = base_dir / 'config-examples' / 'train-params-example.yaml'

    # Initialize the YOLOTrainer
    trainer = ModelTrainer(model_path, hyperparams_path)

    # Run the training process
    trainer.run("Robocup24 Detection Training", "romelavision")