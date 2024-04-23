import cv2
import numpy as np
import threading
import queue
from pathlib import Path
import pyrealsense2 as rs
import torch
import time
from ultralytics import YOLO  # Make sure this import works for your YOLO version

# Define the base directory relative to the script file
base_dir = Path(__file__).resolve().parent.parent

weight_name = 'robocup1-15-tune13'
video_name = 'IMG_0197.MOV'
# Path to the .pt file
model_path = base_dir / 'util' / 'weights' / f'{weight_name}.pt'

# Load yolov8-seg model
model = YOLO(model_path)

# Define path to video file
# source = r"C:\Users\User\Box\RoboCup\Robocup 2023\Test Videos\Vision Test Videos Jul3\Tests_Recordings\Record_raw11.avi"
source = r"C:\Users\User\Box\RoboCup\Robocup 2023\Ethan Videos\July4" + f"\\{video_name}"
cap = cv2.VideoCapture(source)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Source video FPS: {fps}")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can change 'XVID' to 'MP4V' or other compatible codec
output_path = str(base_dir / 'util' / 'video outputs' / f'{video_name}({weight_name}).mp4')  # Change file path or name as needed
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


ret = True
while ret:
    ret, frame = cap.read()
    
    if ret:
        results = model.predict(frame)
        
        # Draw the inference results onto the frame
        frame_ = results[0].plot()
        
        # Write the frame with inference results
        out.write(frame_)
        
        cv2.imshow('Robocup Video Inference', frame_)
        
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
