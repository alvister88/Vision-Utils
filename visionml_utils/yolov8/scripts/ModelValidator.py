import wandb
from ultralytics import YOLO
from pathlib import Path

class ModelValidator:
    def __init__(self, model_weights_path):
        """
        Initializes the ModelValidator with the full path to the model weights.

        Args:
            model_weights_path (str): The full path to the model weights file.
        """
        self.model_path = model_weights_path
        self.model = YOLO(self.model_path)  # Load the model

    def perform_validation(self, data_file, image_size):
        """
        Performs validation using the loaded YOLO model.

        Args:
            data_file (str): The name of the data file to use for validation.
            image_size (int): The dimensions to which the images will be resized.

        Returns:
            list: The results of the model validation.
        """
        return self.model.val(
            data=data_file,
            imgsz=image_size,
            name=f'{Path(self.model_path).stem}-test'
        )

# Usage example
if __name__ == '__main__':
    # Define the paths and parameters outside the class
    base_directory = Path(__file__).resolve().parent
    model_weights_path = base_directory / 'weights' / ''
    data_filename = ''
    image_size = 640

    validator = ModelValidator(model_weights_path)
    results = validator.perform_validation(data_filename, image_size)

