import cv2
from pathlib import Path

class FrameExtractor:
    def __init__(self, video_path, output_dir=None, interval=30, max_frames=None, verbose=False):
        """
        Initializes the FrameExtractor with specific paths for the video file and output directory.

        Args:
            video_path (str): The full path to the video file to process.
            output_dir (str, optional): The directory where the extracted frames will be saved. Defaults to None.
            interval (int, optional): The interval (in frames) at which to save images. Defaults to 30.
            max_frames (int, optional): The maximum number of frames to extract.
            verbose (bool): Whether to print detailed logs. Defaults to False.
        """
        self.video_path = Path(video_path)
        video_name = self.video_path.stem
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / f'{video_name}-extracted_frames'
        self.interval = interval
        self.max_frames = max_frames
        self.verbose = verbose

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"Output directory: {self.output_dir}")

        # Set up video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Error: Unable to open video file {self.video_path}")

        # Get the total number of frames in the video
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.verbose:
            print(f"Successfully opened video file {self.video_path}")
            print(f"Total frames in video: {self.total_frames}")
            print(f"Frame extraction interval: {self.interval}")
            if self.max_frames:
                print(f"Maximum frames to extract: {self.max_frames}")

    def extract_frames(self):
        """
        Extracts frames from the video at the specified interval and saves them as images.
        """
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret or (self.max_frames and saved_count >= self.max_frames):
                if self.verbose:
                    print("End of video or maximum frames reached.")
                break

            if frame_count % self.interval == 0:
                # Construct the output file path
                output_file_path = self.output_dir / f'frame_{saved_count:04d}.jpg'
                # Save the frame as an image
                cv2.imwrite(str(output_file_path), frame)
                if self.verbose:
                    print(f"Saved frame {saved_count} to {output_file_path}")
                saved_count += 1

            frame_count += 1

        # Release the video capture resource
        self.cap.release()
        if self.verbose:
            print("Released video capture resources.")

# Example Usage
if __name__ == '__main__':
    video_path = 'path/to/your/video.mp4'  # Replace with the path to your video file
    output_dir = None  # Replace with the path to your output directory, or None to use the default
    interval = 30  # Replace with the interval at which to save frames (e.g., every 30 frames)
    max_frames = 100  # Replace with the maximum number of frames to extract, or None
    verbose = True  # Set to True to enable verbose logging

    # Create and use the FrameExtractor
    extractor = FrameExtractor(video_path, output_dir, interval, max_frames, verbose)
    extractor.extract_frames()
