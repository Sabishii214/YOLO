import sys
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection with High Accuracy Options")
    parser.add_argument("--source", type=str, default="0", help="Path to video file or '0' for webcam")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to model weights (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (pixels)")
    parser.add_argument("--augment", action="store_true", help="Enable Test Time Augmentation (TTA) for higher accuracy (slower)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config (bytetrack.yaml or botsort.yaml)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--view-img", action="store_true", default=None, help="Show results in a window (Default: True for webcam, False for files)")
    parser.add_argument("--viz-conf", type=float, default=None, help="Visualization threshold (Defaults to match --conf)")
    parser.add_argument("--viz-classes", type=str, nargs="+", default=None, help="Optional: List of classes to show in video (e.g., person car)")
    parser.add_argument("--no-save-txt", action="store_true", help="Disable saving detections to text file (enabled by default)")
    parser.add_argument("--no-metrics", action="store_true", help="Disable performance metrics (metrics enabled by default)")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Setup directories
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Determine input type
    is_webcam = args.source == "0" or args.source.isdigit()

    # Default view_img to True if webcam, else False (unless specified)
    if args.view_img is None:
        args.view_img = is_webcam
    
    # Output filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if is_webcam:
        base_name = "webcam"
        source_path = int(args.source)
    else:
        base_name = Path(args.source).stem
        source_path = args.source
    
    video_output_path = output_folder / f"{base_name}_{timestamp}-output.mp4"
    txt_output_path = output_folder / f"{base_name}_{timestamp}-output.txt"
    report_output_path = output_folder / f"{base_name}_{timestamp}-report.txt"

    # Load Model
    print(f"Loading model: {args.weights}...")
    model = YOLO(args.weights) 
    print("Model loaded successfully.")

    # Open Source
    print(f"Opening source: {source_path}")
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open source {source_path}")
        sys.exit(1)

    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not is_webcam:
        total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        fps = 30 # Default for webcam if not readable
        total_frames_count = -1

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

    # Txt File
    save_txt = not args.no_save_txt
    if save_txt:
        txt_file = open(txt_output_path, "w")
        txt_file.write("frame_no,detection_no,class,x,y,width,height,confidence\n")
    else:
        txt_file = None

    print(f"\nStarting detection...")
    print(f"Config: Conf={args.conf}, IoU={args.iou}, ImgSz={args.imgsz}, Augment={args.augment}")
    print("Press Q or ESC to stop early.\n")

    frame_no = 0
    start_time = time.time()
    
    # Performance tracking
    show_metrics = not args.no_metrics
    if show_metrics:
        inference_times = []
        confidence_scores = []
        detection_counts = []
        class_detections = defaultdict(int)
        class_confidences = defaultdict(list)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_no += 1
            
            # Run Tracking/Detection
            # persist=True is important for tracking video streams
            inference_start = time.time() if show_metrics else None
            
            results = model.track(
                frame, 
                persist=True, 
                conf=args.conf, 
                iou=args.iou, 
                imgsz=args.imgsz, 
                augment=args.augment,
                tracker=args.tracker,
                verbose=False
            )

            if show_metrics:
                inference_times.append((time.time() - inference_start) * 1000)  # ms

            result = results[0]
            
            # Collect metrics
            if show_metrics:
                num_dets = len(result.boxes)
                detection_counts.append(num_dets)
                if num_dets > 0:
                    for i, box in enumerate(result.boxes):
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        
                        confidence_scores.append(conf)
                        class_detections[class_name] += 1
                        class_confidences[class_name].append(conf)
            
            # Visualization
            if is_webcam:
                # Default style for webcam (no custom changes)
                annotated_frame = result.plot()
            else:
                # Custom "Thin Green & Transparent" style for video files
                annotated_frame = frame.copy()
                if result.boxes:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = model.names[cls]
                        
                        # Filter by confidence AND classes (if specified)
                        # Use base --conf if --viz-conf is not set
                        v_conf = args.viz_conf if args.viz_conf is not None else args.conf
                        show_box = conf >= v_conf
                        
                        if args.viz_classes and name not in args.viz_classes:
                            show_box = False
                            
                        if show_box:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            
                            # 1. Draw green outline (thickness 2 for better visibility)
                            green = (0, 255, 0)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), green, 2)
                            
                            # 2. Draw semi-transparent fill
                            overlay = annotated_frame.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), green, -1)
                            alpha = 0.15 # Transparency factor
                            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
                            
                            # 3. Draw clean label
                            label = f"{name} {conf:.2f}"
                            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            
                            # Label background
                            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), green, -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            out_video.write(annotated_frame)
            
            if args.view_img:
                cv2.imshow("YOLOv8 Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            # Save Text
            if txt_file and result.boxes:
                for i, box in enumerate(result.boxes):
                    # Using detection index (1-based) as detection_no
                    detection_no = i + 1
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    
                    txt_file.write(f"{frame_no},{detection_no},{name},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{conf:.4f}\n")

            # Progress log
            if frame_no % 50 == 0:
                elapsed = time.time() - start_time
                curr_fps = frame_no / elapsed
                progress = f"{frame_no}/{total_frames_count}" if total_frames_count > 0 else f"{frame_no}"
                print(f"Processing frame {progress} | FPS: {curr_fps:.2f} | Detections: {len(result.boxes)}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nDetection stopped by user.")

    cap.release()
    out_video.release()
    if txt_file:
        txt_file.close()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    
    print(f"\nProcessing complete.")
    print(f"Output Video: {video_output_path}")
    
    # Print and Save performance metrics
    if show_metrics and frame_no > 0:
        avg_fps = frame_no / total_time
        avg_inference = np.mean(inference_times)
        std_inference = np.std(inference_times)
        
        report_lines = []
        report_lines.append(f"{'='*70}")
        report_lines.append(f"PERFORMANCE REPORT - {base_name}")
        report_lines.append(f"{'='*70}\n")
        
        report_lines.append(f"SPEED METRICS:")
        report_lines.append(f"   Total Frames: {frame_no}")
        report_lines.append(f"   Total Time: {total_time:.2f}s")
        report_lines.append(f"   Average FPS: {avg_fps:.2f}")
        report_lines.append(f"   Avg Inference: {avg_inference:.2f}ms +/- {std_inference:.2f}ms")
        report_lines.append(f"   Min Inference: {np.min(inference_times):.2f}ms")
        report_lines.append(f"   Max Inference: {np.max(inference_times):.2f}ms")
        
        if confidence_scores:
            avg_conf = np.mean(confidence_scores)
            std_conf = np.std(confidence_scores)
            total_dets = sum(detection_counts)
            avg_dets = np.mean(detection_counts)
            
            report_lines.append(f"\nDETECTION METRICS:")
            report_lines.append(f"   Total Detections: {total_dets}")
            report_lines.append(f"   Avg Detections/Frame: {avg_dets:.2f}")
            report_lines.append(f"   Avg Confidence: {avg_conf:.3f} ({avg_conf*100:.1f}%)")
            report_lines.append(f"   Confidence Std Dev: {std_conf:.3f}")
            
            # Confidence distribution
            high_conf = sum(1 for c in confidence_scores if c >= 0.7)
            med_conf = sum(1 for c in confidence_scores if 0.4 <= c < 0.7)
            low_conf = sum(1 for c in confidence_scores if c < 0.4)
            
            report_lines.append(f"\nCONFIDENCE DISTRIBUTION:")
            report_lines.append(f"   High (>=0.7): {high_conf} ({high_conf/len(confidence_scores)*100:.1f}%)")
            report_lines.append(f"   Medium (0.4-0.7): {med_conf} ({med_conf/len(confidence_scores)*100:.1f}%)")
            report_lines.append(f"   Low (<0.4): {low_conf} ({low_conf/len(confidence_scores)*100:.1f}%)")
            
            # Top classes
            if class_detections:
                report_lines.append(f"\nTOP DETECTED CLASSES:")
                sorted_classes = sorted(class_detections.items(), key=lambda x: x[1], reverse=True)
                for class_name, count in sorted_classes[:5]:
                    avg_class_conf = np.mean(class_confidences[class_name])
                    report_lines.append(f"   {class_name:15s}: {count:5d} | Avg Conf: {avg_class_conf:.3f}")
        
        report_lines.append(f"\n{'='*70}\n")
        
        # Output to terminal
        report_content = "\n".join(report_lines)
        print(f"\n{report_content}")
        
        # Output to file
        with open(report_output_path, "w") as rf:
            rf.write(report_content)
        print(f"Performance Report saved to: {report_output_path}")


if __name__ == "__main__":
    main()
