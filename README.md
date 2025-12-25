
# Project Title

A brief description of what this project does and who it's for

YOLO Object Detection(YOLOv8)

This project performs **object detection on a video** using **YOLOv8 (Ultralytics)** inside a **Docker container**.  
It processes an input video, detects objects frame-by-frame, and generates:

-  An **annotated output video**
-  A **text file containing detection details**

The entire pipeline runs inside Docker for **easy setup and reproducibility**.

---

## Project Structure

```text
YOLO/
├── docker/
│   ├── Dockerfile
│   └── build-docker.sh
│
├── scripts/
│   ├── pull-docker.sh
│   ├── run-docker.sh
│   └── run-detector.sh
│
├── input/
│   └── input-video.mp4
│
├── output/
│   ├── output-video.mp4
│   └── output-video.txt
│
├── git/
│   └── git-info.txt
│
└── README.md
```
## What This Project Does

1. Reads a video from the `input/` folder  
2. Runs **YOLOv8n** (nano model) on every frame  
3. Draws bounding boxes and labels on detected objects  
4. Saves:
   - Annotated video → `output/output-video.mp4`
   - Detection data → `output/output-video.txt`

---

## Requirements

Make sure you have:

- **Docker Desktop installed and running**
- **VS Code (recommended)**
- **Windows / macOS / Linux**

> No Python installation needed on host (Docker handles everything)

---
## Example Output (Terminal)

```text
Input video found
Loading YOLOv8 model...
Processing video...

Video info: 1920x1080 @ 25 FPS, 341 frames

Progress: 300/341 frames (88.0%)

Processing Complete!
Total frames processed: 341
Total objects detected: 11119
```

---

## Output Files

After completion, check:

```text
output/
├── output-video.mp4   # Video with bounding boxes
└── output-video.txt   # Detection details
```

### Output Text File Format

```text
frame_no,detection_no,class,x,y,width,height,confidence
1,1,person,245,123,89,234,0.8923
1,2,car,456,234,156,98,0.9234
```

**Column Meaning:**

| Column | Description |
|------|------------|
| frame_no | Frame number |
| detection_no | Detection index |
| class | Object class |
| x, y | Top-left corner |
| width, height | Bounding box size |
| confidence | Detection confidence |

---

## Model Information

- **Model:** YOLOv8n (Nano)
- **Framework:** Ultralytics YOLOv8
- **Installed via:** `pip install ultralytics`
- **Weights:** Auto-downloaded on first run

---

## Docker Details

- Base Image: `python:3.9`
- Packages installed:
  - `ultralytics`
  - `opencv-python-headless`
  - `pandas`
- No GPU required (CPU-only)

---

## Notes

- First run may take longer (model download)
- Processing time depends on video length and resolution
- Output folder is created automatically if missing

---