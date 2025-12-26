#!/bin/bash

# Build Docker image
docker build -t yolo-detector:latest -f Dockerfile .

echo "Docker image built successfully!"
echo "Image name: yolo-detector:latest"