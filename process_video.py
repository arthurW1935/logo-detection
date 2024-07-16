import av
import tqdm
import torch
import ultralytics
import argparse
import json

detection_time = [[],[]]

def process_video(video_path, batch_size=16, frame_skip=1):
    container = av.open(video_path)
    frames = []
    timestamps = []
    frame_count = 0

    for frame in tqdm.tqdm(container.decode(video=0)):
        if frame_count % frame_skip == 0:
            frame_array = frame.to_ndarray(format='bgr24')
            frames.append(frame_array)
            timestamps.append(round(frame.time, 2))
            
            if len(frames) == batch_size:
                process_batch(frames, timestamps)
                frames = []
                timestamps = []

        frame_count += 1

    if frames:
        process_batch(frames, timestamps)

def process_batch(frames, timestamps):
    results = model(frames, stream=True, verbose=False)
    
    for res, timestamp in zip(results, timestamps):
        result_classes = res.boxes.cls.tolist()
        if 0 in result_classes:
            detection_time[0].append(timestamp)
        if 1 in result_classes:
            detection_time[1].append(timestamp)

def save_results_to_json(detection_time, output_path):
    detection_obj = {
        "Pepsi_pts"    : detection_time[1],
        "CocaCols_pts" : detection_time[0], 
    }
    with open(output_path, 'w') as json_file:
        json.dump(detection_obj, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video for object detection.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing frames')
    parser.add_argument('--frame_skip', type=int, default=1, help='Number of frames to skip')
    parser.add_argument('--output_json', type=str, default='result.json', help='Path to the output JSON file')

    args = parser.parse_args()
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading the latest model...")
    model = ultralytics.YOLO('logo-detection-latest.pt').to(device)
    print("Model loaded successfully!")

    print("Starting video processing...")
    process_video(args.video_path, batch_size=args.batch_size, frame_skip=args.frame_skip)
    print("Video processing complete!")

    save_results_to_json(detection_time, args.output_json)
    print(f"Detection times saved to {args.output_json}")
