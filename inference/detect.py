"""
Run Vehicle Detection on Video
================================
Simple script to detect vehicles in a video using the trained model.
Outputs results as JSON.

Usage:
    python detect.py --video ../data/raw_videos/sample.avi
    python detect.py --video ../data/raw_videos/sample.avi --output results.json

Author: Richa Mathur
"""

import argparse
import json
import sys
import time
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


def detect_vehicles(video_path, confidence=0.3, frame_interval=5, output_path=None):
    """Run detection on a video and return/save results."""
    video_path = Path(video_path)
    
    # Load model
    if TRAINED_MODEL.exists():
        model = YOLO(str(TRAINED_MODEL))
        class_map = CUSTOM_CLASSES
        model_name = "fine-tuned"
    else:
        model = YOLO(PRETRAINED_MODEL)
        class_map = VEHICLE_CLASSES
        model_name = "pre-trained"
    
    print(f"  Model: {model_name} | Video: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Could not open: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_time = time.time()
    all_detections = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
        
        results = model(frame, conf=confidence, verbose=False)
        
        frame_dets = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if TRAINED_MODEL.exists() or cls_id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    frame_dets.append({
                        "class": class_map.get(cls_id, "unknown"),
                        "confidence": round(float(boxes.conf[i]), 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    })
        
        all_detections.append({
            "frame": frame_count,
            "time_sec": round(frame_count / fps, 2),
            "vehicles": frame_dets,
            "count": len(frame_dets),
        })
    
    cap.release()
    elapsed = round(time.time() - start_time, 2)
    
    total_vehicles = sum(d["count"] for d in all_detections)
    
    result = {
        "video": video_path.name,
        "model": model_name,
        "processing_time_sec": elapsed,
        "frames_processed": len(all_detections),
        "total_vehicles_detected": total_vehicles,
        "avg_per_frame": round(total_vehicles / max(len(all_detections), 1), 1),
        "detections": all_detections,
    }
    
    # Save or print
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  ✓ Results saved: {output_path}")
    
    print(f"  {total_vehicles} vehicles in {len(all_detections)} frames ({elapsed}s)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Detect vehicles in video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    
    detect_vehicles(args.video, args.confidence, args.interval, args.output)


if __name__ == "__main__":
    main()
