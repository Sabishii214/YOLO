#!/bin/bash

echo "Starting YOLO Object Detection Container"

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"
echo ""

# Check if input video exists
if [ ! -f "$PROJECT_DIR/input/input-video.mp4" ]; then
    echo "ERROR: Input video not found!"
    echo "Please place a video file at: $PROJECT_DIR/input/input-video.mp4"
    exit 1
fi

echo "Input video found"
echo "Starting Docker container..."
echo ""

# Run Docker container
docker run -it --rm \
    -v "$PROJECT_DIR/app" \
    yolo-detector:latest \
    /app/scripts/run-detector.sh