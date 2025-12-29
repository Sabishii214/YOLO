#!/bin/bash

# Default values
INPUT_VIDEO=""
MODEL="yolov8l.pt" 
CONF="0.3"
IOU="0.5"
IMGSZ="1280"
TRACKER="bytetrack.yaml"

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: ./run-docker.sh [input_video_path]"
    echo "  input_video_path: Path to video (relative to project root). Default: Batch process all videos in input/ folder"
    exit 0
fi

if [ -n "$1" ]; then
    INPUT_VIDEO="$1"
fi

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "Starting High-Accuracy Video Detection in Docker"
echo "Input: $INPUT_VIDEO"
echo "Model: $MODEL"
echo "TTA: Enabled (Test Time Augmentation)"
echo "Tracker: $TRACKER"
echo ""

# Run Docker
docker run --rm -it \
    -v "$PROJECT_DIR:/app" \
    --ipc=host \
    yolo-detector:latest \
    /app/scripts/run-detector.sh \
    --source "$INPUT_VIDEO" \
    --weights "$MODEL" \
    --conf "$CONF" \
    --iou "$IOU" \
    --imgsz "$IMGSZ" \
    --augment \
    --tracker "$TRACKER" \
    --output-dir "output"