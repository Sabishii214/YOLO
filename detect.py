import sys
from pathlib import Path
import cv2
from ultralytics import YOLO
import time

output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

video_output_path = output_folder / "webcam-output.mp4"
txt_output_path = output_folder / "webcam-output.txt"

model_path = "yolov8n.pt"

model = YOLO(model_path)
print("Model loaded successfully\n")

print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam")
    sys.exit(1)

# Get webcam properties
fps = 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

# Open text file for detections with proper headers
txt_file = open(txt_output_path, "w")
txt_file.write("frame_no,detection_no,class,x,y,width,height,confidence\n")

print("Starting detection loop (press Q or ESC to quit)\n")

frame_no = 0
total_detections = 0
progress_interval = 30 
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame_no += 1

    # Run YOLOv8 detection
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

            # Write to CSV with frame_no and detection_no
            txt_file.write(f"{frame_no},{detection_no},{name},{x},{y},{w},{h},{conf:.4f}\n")

    # Write video frame
    out_video.write(frame)
    cv2.imshow("YOLOv8 Webcam", frame)

    # Live progress print
    if frame_no % progress_interval == 0:
        elapsed = time.time() - start_time
        print(f"Progress: {frame_no} frames, {total_detections} detections, elapsed {elapsed:.1f}s")

    # Quit on 'q' or ESC
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
out_video.release()
txt_file.close()
cv2.destroyAllWindows()

print("\nDetection completed")
print(f"Total frames processed: {frame_no}")
print(f"Total objects detected: {total_detections}")
print(f"Video saved to: {video_output_path}")
print(f"Detections saved to: {txt_output_path}")
