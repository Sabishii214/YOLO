#!/bin/bash

# Build Docker image
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

docker build -t yolo-detector:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_DIR"

echo "Docker image built successfully!"
echo "Image name: yolo-detector:latest"