#!/bin/bash

# Create writable YOLO config directory
mkdir -p /app/config
export YOLO_CONFIG_DIR=/app/config

python3 << 'EOF'

import sys
from pathlib import Path
import cv2
from ultralytics import YOLO
import time

input_folder = Path("input")
output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

model_path = "yolov8n.pt"

model = YOLO(model_path)
print("Model loaded successfully\n")

video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
video_paths = [p for p in input_folder.glob("*") if p.suffix.lower() in video_extensions]

if not video_paths:
    print(f"No videos found in {input_folder}")
    sys.exit(1)

print(f"Found {len(video_paths)} video(s) to process.\n")

for video_path in video_paths:
    print(f"Processing: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        continue

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = output_folder / f"{video_path.stem}-output.mp4"
    out_video = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # TXT output
    txt_output_path = output_folder / f"{video_path.stem}-output.txt"
    txt_file = open(txt_output_path, "w")
    txt_file.write("frame_no,detection_no,class,x,y,width,height,confidence\n")

    frame_no = 0
    total_detections = 0
    progress_interval = 30  # print progress every 30 frames
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        # YOLO detection
        results = model(frame, verbose=False)
        detection_no = 0

        for result in results:
            for box in result.boxes:
                detection_no += 1
                total_detections += 1

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]

                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Write to CSV
                txt_file.write(f"{frame_no},{detection_no},{name},{x},{y},{w},{h},{conf:.4f}\n")

        out_video.write(frame)

        # Live progress print
        if frame_no % progress_interval == 0:
            elapsed = time.time() - start_time
            progress_pct = (frame_no / total_frames) * 100
            print(f"Progress: {frame_no}/{total_frames} frames ({progress_pct:.1f}%), {total_detections} detections, elapsed {elapsed:.1f}s")

    cap.release()
    out_video.release()
    txt_file.close()

    print(f"Finished {video_path.name}: {frame_no} frames, {total_detections} objects detected\n")

print("=== All videos processed successfully ===")
