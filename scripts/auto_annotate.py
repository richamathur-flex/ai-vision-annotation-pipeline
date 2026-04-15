"""
Auto-Annotate Traffic Videos using YOLOv8
==========================================
This script runs YOLOv8 object detection on toll road surveillance videos
and generates COCO-format JSON annotations.

What this script does (step by step):
1. Takes a video file (MP4) as input
2. Extracts frames at a set interval (e.g., every 5th frame)
3. Runs YOLOv8 to detect vehicles in each frame
4. Saves the detections as COCO JSON format
5. Also saves annotated frame images with bounding boxes drawn

Why we need this:
- Manual annotation in CVAT is slow (good for 2-3 videos to prove skills)
- This script auto-detects vehicles so we only need to REVIEW + ADD attributes
  (axle count, direction, lane — things YOLO can't detect)
- Shows Python + CV skills on GitHub

Usage:
    python auto_annotate.py --video data/converted/0101-1_20240307173236.mp4
    python auto_annotate.py --video-dir data/converted/ --all
    python auto_annotate.py --video data/converted/sample.mp4 --save-frames

Requirements:
    pip install ultralytics opencv-python

Author: Richa Mathur
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2

# ── Try importing ultralytics (YOLOv8) ──────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("=" * 60)
    print("ERROR: ultralytics not installed!")
    print("Run: pip install ultralytics")
    print("=" * 60)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — Change these values for your project
# ═══════════════════════════════════════════════════════════════════════

# Which YOLO model to use
# Options: "yolov8n.pt" (nano/fast), "yolov8s.pt" (small), "yolov8m.pt" (medium)
# For this project, nano is fine — we're showing the pipeline, not competing on accuracy
YOLO_MODEL = "yolov8n.pt"

# How often to extract frames from video
# 5 = every 5th frame. For 30fps video, this gives 6 frames per second
# Lower number = more frames = more annotations = slower
FRAME_INTERVAL = 5

# Minimum confidence score to keep a detection (0.0 to 1.0)
# 0.3 = keep detections YOLO is at least 30% sure about
# Lower = more detections (including false positives)
# Higher = fewer but more reliable detections
CONFIDENCE_THRESHOLD = 0.3

# COCO vehicle class IDs from the COCO dataset that YOLO was trained on
# We only care about vehicles, not people/animals/furniture
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Our custom category mapping for the toll road project
# These are the categories in our COCO JSON output
CUSTOM_CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": "vehicle"},
    {"id": 2, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 3, "name": "bus", "supercategory": "vehicle"},
    {"id": 4, "name": "truck", "supercategory": "vehicle"},
]

# Map YOLO's COCO class IDs to our custom category IDs
YOLO_TO_CUSTOM = {
    2: 1,   # YOLO "car" → our category 1
    3: 2,   # YOLO "motorcycle" → our category 2
    5: 3,   # YOLO "bus" → our category 3
    7: 4,   # YOLO "truck" → our category 4
}


# ═══════════════════════════════════════════════════════════════════════
# COCO JSON BUILDER
# ═══════════════════════════════════════════════════════════════════════

class COCOAnnotationBuilder:
    """
    Builds a COCO-format JSON annotation file.

    COCO format has 3 main sections:
    - "images": list of all frames (id, file_name, width, height)
    - "annotations": list of all detections (bounding box, category, etc.)
    - "categories": list of object types (car, truck, bus, motorcycle)

    This is the SAME format you exported from CVAT at Oracle!
    """

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = CUSTOM_CATEGORIES
        self.image_id_counter = 0
        self.annotation_id_counter = 0

    def add_image(self, file_name, width, height):
        """Register a new frame image."""
        self.image_id_counter += 1
        image_entry = {
            "id": self.image_id_counter,
            "file_name": file_name,
            "width": width,
            "height": height,
        }
        self.images.append(image_entry)
        return self.image_id_counter

    def add_annotation(self, image_id, category_id, bbox, confidence, segmentation=None):
        """
        Add a detection annotation for one vehicle.

        Parameters:
            image_id: which frame this detection belongs to
            category_id: 1=car, 2=motorcycle, 3=bus, 4=truck
            bbox: [x, y, width, height] — top-left corner + dimensions
            confidence: YOLO's confidence score (0.0 to 1.0)
            segmentation: polygon points (optional — YOLO gives boxes, not polygons)

        The "attributes" field contains placeholders for the data that
        YOLO can't detect — you add these manually in CVAT:
            - axle_count: how many axles (determines toll price)
            - direction: forward or backward
            - lane: which highway lane
            - is_occluded: partially hidden?
            - is_truncated: cut off at frame edge?
        """
        self.annotation_id_counter += 1

        # Calculate area from bounding box
        area = bbox[2] * bbox[3]  # width * height

        # If no polygon segmentation provided, create one from the bounding box
        # This creates a rectangle polygon: [x1,y1, x2,y1, x2,y2, x1,y2]
        if segmentation is None:
            x, y, w, h = bbox
            segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]

        annotation = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [round(v, 2) for v in bbox],
            "area": round(area, 2),
            "segmentation": segmentation,
            "iscrowd": 0,
            "confidence": round(confidence, 4),
            # ── These attributes need manual review in CVAT ──
            # YOLO detects WHERE vehicles are, but can't determine:
            "attributes": {
                "axle_count": -1,       # -1 means "not yet labeled"
                "direction": "unknown",  # "forward" or "backward"
                "lane": -1,             # which lane (0-20)
                "is_occluded": False,   # partially hidden?
                "is_truncated": False,  # cut off at edge?
                "group_id": 0,          # for truck+trailer grouping
                "needs_review": True,   # flag for manual review
            },
        }
        self.annotations.append(annotation)
        return self.annotation_id_counter

    def to_dict(self):
        """Convert to COCO JSON dictionary."""
        return {
            "info": {
                "description": "Auto-annotated toll road vehicle detection",
                "version": "1.0",
                "year": 2026,
                "contributor": "Richa Mathur",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "url": "https://github.com/richamathur-flex/ai-vision-annotation-pipeline",
            },
            "licenses": [
                {"id": 1, "name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"}
            ],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save(self, output_path):
        """Save COCO JSON to file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  ✓ Saved COCO JSON: {output_path}")
        print(f"    → {len(self.images)} frames, {len(self.annotations)} detections")


# ═══════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR
# ═══════════════════════════════════════════════════════════════════════

def process_video(
    video_path,
    output_dir="data/annotations",
    frames_dir=None,
    model=None,
    frame_interval=FRAME_INTERVAL,
    confidence=CONFIDENCE_THRESHOLD,
    save_frames=False,
):
    """
    Process a single video file:
    1. Open video with OpenCV
    2. Extract frames at interval
    3. Run YOLOv8 on each frame
    4. Build COCO JSON with all detections
    5. Optionally save annotated frame images

    Parameters:
        video_path: path to the MP4 video file
        output_dir: where to save COCO JSON
        frames_dir: where to save extracted frame images
        model: loaded YOLO model (if None, will load)
        frame_interval: extract every Nth frame
        confidence: minimum detection confidence
        save_frames: if True, save frames with drawn bounding boxes
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  ✗ Video not found: {video_path}")
        return None

    # ── Load YOLO model if not provided ──
    if model is None:
        print(f"  Loading YOLO model: {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)

    # ── Open video ──
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Could not open video: {video_path}")
        return None

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = video_path.stem

    print(f"\n{'='*60}")
    print(f"  Processing: {video_path.name}")
    print(f"  Resolution: {width}x{height} | FPS: {fps:.1f} | Frames: {total_frames}")
    print(f"  Extracting every {frame_interval}th frame → ~{total_frames // frame_interval} frames")
    print(f"{'='*60}")

    # ── Create output directories ──
    os.makedirs(output_dir, exist_ok=True)
    if save_frames:
        if frames_dir is None:
            frames_dir = f"data/sample_frames/{video_name}"
        os.makedirs(frames_dir, exist_ok=True)

    # ── Initialize COCO builder ──
    coco = COCOAnnotationBuilder()

    # ── Process frames ──
    frame_count = 0
    processed_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames based on interval
        if frame_count % frame_interval != 0:
            continue

        processed_count += 1
        frame_filename = f"{video_name}_frame_{frame_count:06d}.png"

        # ── Run YOLO detection ──
        results = model(frame, conf=confidence, verbose=False)

        # ── Register frame in COCO ──
        image_id = coco.add_image(frame_filename, width, height)

        # ── Process each detection ──
        frame_detections = 0
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Get class ID and check if it's a vehicle
                class_id = int(boxes.cls[i])
                if class_id not in VEHICLE_CLASSES:
                    continue

                # Get bounding box [x1, y1, x2, y2] → convert to [x, y, w, h]
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO format: [x, y, width, height]

                # Get confidence score
                conf = float(boxes.conf[i])

                # Map YOLO class to our custom category
                custom_category = YOLO_TO_CUSTOM.get(class_id, 1)

                # Add to COCO annotations
                coco.add_annotation(image_id, custom_category, bbox, conf)
                frame_detections += 1

                # Draw bounding box on frame (if saving frames)
                if save_frames:
                    label = f"{VEHICLE_CLASSES[class_id]} {conf:.2f}"
                    color = {2: (0, 255, 0), 3: (255, 165, 0), 5: (255, 0, 0), 7: (0, 0, 255)}
                    box_color = color.get(class_id, (0, 255, 0))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    cv2.putText(
                        frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2,
                    )

        total_detections += frame_detections

        # Save annotated frame image
        if save_frames:
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

        # Progress indicator
        if processed_count % 20 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"    Frame {frame_count}/{total_frames} ({pct:.0f}%) — {frame_detections} vehicles")

    cap.release()

    # ── Save COCO JSON ──
    output_path = os.path.join(output_dir, f"{video_name}_annotations.json")
    coco.save(output_path)

    # ── Print summary ──
    print(f"\n  Summary for {video_path.name}:")
    print(f"    Frames processed: {processed_count}")
    print(f"    Total detections: {total_detections}")
    print(f"    Avg vehicles/frame: {total_detections / max(processed_count, 1):.1f}")
    if save_frames:
        print(f"    Annotated frames saved to: {frames_dir}/")

    return output_path


# ═══════════════════════════════════════════════════════════════════════
# MAIN — Run from command line
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate toll road videos using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate a single video
  python auto_annotate.py --video data/converted/0101-1_20240307173236.mp4

  # Annotate and save frame images with bounding boxes
  python auto_annotate.py --video data/converted/sample.mp4 --save-frames

  # Annotate all videos in a directory
  python auto_annotate.py --video-dir data/converted/ --all

  # Use a bigger YOLO model for better accuracy
  python auto_annotate.py --video sample.mp4 --model yolov8m.pt

  # Adjust confidence threshold
  python auto_annotate.py --video sample.mp4 --confidence 0.5
        """,
    )

    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video-dir", type=str, help="Directory containing videos")
    parser.add_argument("--all", action="store_true", help="Process all videos in directory")
    parser.add_argument("--output", type=str, default="data/annotations", help="Output directory for COCO JSON")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frame images")
    parser.add_argument("--model", type=str, default=YOLO_MODEL, help="YOLO model to use")
    parser.add_argument("--interval", type=int, default=FRAME_INTERVAL, help="Extract every Nth frame")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, help="Min confidence threshold")

    args = parser.parse_args()

    # ── Validate inputs ──
    if not args.video and not args.video_dir:
        parser.print_help()
        print("\n  Error: Provide --video or --video-dir")
        sys.exit(1)

    # ── Load YOLO model once (reuse across videos) ──
    print(f"\n  Loading YOLOv8 model: {args.model}")
    print(f"  (First run will download the model — ~6MB for nano)\n")
    model = YOLO(args.model)

    # ── Process single video ──
    if args.video:
        process_video(
            args.video,
            output_dir=args.output,
            model=model,
            frame_interval=args.interval,
            confidence=args.confidence,
            save_frames=args.save_frames,
        )

    # ── Process all videos in directory ──
    elif args.video_dir and args.all:
        video_dir = Path(args.video_dir)
        video_files = sorted(
            list(video_dir.glob("*.mp4"))
            + list(video_dir.glob("*.avi"))
            + list(video_dir.glob("*.mov"))
        )

        if not video_files:
            print(f"  No video files found in {video_dir}")
            sys.exit(1)

        print(f"  Found {len(video_files)} videos in {video_dir}")
        print(f"  {'─' * 40}")

        for i, vf in enumerate(video_files, 1):
            print(f"\n  [{i}/{len(video_files)}]")
            process_video(
                vf,
                output_dir=args.output,
                model=model,
                frame_interval=args.interval,
                confidence=args.confidence,
                save_frames=args.save_frames,
            )

        print(f"\n{'='*60}")
        print(f"  Done! Processed {len(video_files)} videos.")
        print(f"  COCO JSON files saved to: {args.output}/")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
