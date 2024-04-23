import cv2
from pathlib import Path
from ultralytics import YOLO  # Adjust this import according to your YOLO version

class VideoProcessor:
    def __init__(self, model_weights_path, video_path, output_path):
        """
        Initializes the VideoProcessor with specific paths for model weights and video file.

        Args:
            model_weights_path (str): The full path to the model weights file.
            video_path (str): The full path to the video file to process.
            output_path (str): The full path where the processed video will be saved.
        """
        self.model_path = model_weights_path
        self.video_path = video_path
        self.output_path = output_path

        # Load YOLO model
        self.model = YOLO(self.model_path)

        # Set up video capture and writer
        self.cap = cv2.VideoCapture(self.video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

    def process_video(self):
        """
        Processes the video, performing inference on each frame and writing the results to an output file.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Perform inference
            results = self.model.predict(frame)
            
            # Draw the inference results onto the frame
            frame_with_results = results[0].plot()
            
            # Write the frame with inference results
            self.out.write(frame_with_results)
            
            # Optionally display the frame (comment out if running in non-interactive mode)
            cv2.imshow('Robocup Video Inference', frame_with_results)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Release resources
        self.release_resources()

    def release_resources(self):
        """
        Releases video capture and writer resources and closes any open windows.
        """
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

# Example Usage
if __name__ == '__main__':
    # Define base directory and other parameters
    base_dir = Path(__file__).resolve().parent
    model_weights_path = str(base_dir / 'weights' / 'robocup1-15-tune13.pt')
    video_path = str(base_dir / 'videos' / 'IMG_0197.MOV')
    output_path = str(base_dir / 'video outputs' / 'IMG_0197(robocup1-15-tune13).mp4')

    # Create and use the VideoProcessor
    processor = VideoProcessor(model_weights_path, video_path, output_path)
    processor.process_video()
