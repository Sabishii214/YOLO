#!/bin/bash

# Ensure we are in the app root
cd /app

# Process ALL videos in the input/ directory using High-Accuracy settings.
if [ $# -eq 0 ]; then
    echo "No arguments provided. Running in Batch Mode (Legacy Behavior)."
    echo "Scanning input/ directory..."
    
    # Enable nullglob so wildcard expands to empty string if no matches
    shopt -s nullglob
    files=(input/*.mp4 input/*.avi input/*.mov input/*.mkv)
    
    if [ ${#files[@]} -eq 0 ]; then
        echo "No video files found in input/ directory."
        exit 1
    fi

    for video in "${files[@]}"; do
        echo ""
        echo "Processing: $video"
        echo ""
        python3 detect.py \
            --source "$video" \
            --weights yolov8l.pt \
            --augment \
            --tracker bytetrack.yaml
    done
    echo "Batch processing complete."
else
    python3 detect.py "$@"
fi