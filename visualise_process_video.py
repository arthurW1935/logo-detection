import cv2
from ultralytics import YOLO
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Process a video file with YOLOv8.')
parser.add_argument('video_path', type=str, help='Path to the video file')

args = parser.parse_args()

model = YOLO("logo-detection-latest.pt")

cap = cv2.VideoCapture(args.video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
