"""
Draw Bounding Boxes on Video Frames
=====================================
Takes detection results and draws annotated overlays on frames.

Similar to Ashutosh's DrawBoxesPostApiCalls_Helper.py from the
Oracle project — but using our YOLO model instead of OCI Vision API.

Usage:
    python draw_boxes.py --video ../data/raw_videos/sample.avi --output ../data/annotated/
    python draw_boxes.py --video ../data/raw_videos/sample.avi --show  (live preview)

Author: Richa Mathur
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Run: pip install ultralytics")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
TRAINED_MODEL = PROJECT_ROOT / "models" / "weights" / "best.pt"
PRETRAINED_MODEL = "yolov8n.pt"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CUSTOM_CLASSES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}

COLORS = {
    "car": (0, 200, 0),
    "motorcycle": (255, 165, 0),
    "bus": (200, 0, 0),
    "truck": (0, 0, 200),
}


def draw_detections_on_video(video_path, output_dir=None, show=False,
                              confidence=0.3, model=None):
    """
    Run YOLO on video and draw bounding boxes on every frame.
    Saves as a new annotated video file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  ✗ Video not found: {video_path}")
        return
    
    # Load model
    if model is None:
        if TRAINED_MODEL.exists():
            print(f"  Using fine-tuned model: {TRAINED_MODEL}")
            model = YOLO(str(TRAINED_MODEL))
            using_custom = True
        else:
            print(f"  Using pre-trained model: {PRETRAINED_MODEL}")
            model = YOLO(PRETRAINED_MODEL)
            using_custom = False
    else:
        using_custom = TRAINED_MODEL.exists()
    
    class_map = CUSTOM_CLASSES if using_custom else VEHICLE_CLASSES
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Could not open: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n  Processing: {video_path.name} ({w}x{h}, {fps:.0f}fps, {total} frames)")
    
    # Setup output video writer
    writer = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{video_path.stem}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    frame_count = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        results = model(frame, conf=confidence, verbose=False)
        
        frame_dets = 0
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                
                if not using_custom and class_id not in VEHICLE_CLASSES:
                    continue
                
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                conf = float(boxes.conf[i])
                class_name = class_map.get(class_id, "vehicle")
                color = COLORS.get(class_name, (0, 255, 0))
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{class_name} {conf:.0%}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                frame_dets += 1
        
        total_detections += frame_dets
        
        # Add frame info overlay
        info = f"Frame {frame_count}/{total} | Vehicles: {frame_dets}"
        cv2.putText(frame, info, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Write or show
        if writer:
            writer.write(frame)
        
        if show:
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
    if writer:
        writer.release()
        print(f"  ✓ Saved annotated video: {out_path}")
    if show:
        cv2.destroyAllWindows()
    
    print(f"  Total detections: {total_detections} across {frame_count} frames")
    print(f"  Avg: {total_detections / max(frame_count, 1):.1f} vehicles/frame")


def main():
    parser = argparse.ArgumentParser(description="Draw bounding boxes on video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="data/annotated", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--show", action="store_true", help="Show live preview")
    args = parser.parse_args()
    
    draw_detections_on_video(args.video, args.output, args.show, args.confidence)


if __name__ == "__main__":
    main()
