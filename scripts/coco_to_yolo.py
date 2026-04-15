"""
Convert COCO JSON annotations to YOLO format
=============================================
COCO format:  JSON file with bounding boxes as [x, y, width, height] in pixels
YOLO format:  One .txt file per image with boxes as [class x_center y_center width height] normalized (0-1)

Why we need this:
- Our auto_annotate.py outputs COCO JSON (industry standard)
- YOLOv8 training requires YOLO format text files
- This script bridges the two formats

Usage:
    python coco_to_yolo.py --coco data/annotations/video_annotations.json --output data/yolo_labels/

Author: Richa Mathur
"""

import argparse
import json
import os
import sys
from pathlib import Path


def convert_coco_to_yolo(coco_json_path, output_dir):
    """
    Convert a COCO JSON annotation file to YOLO format text files.

    COCO bbox format: [x_top_left, y_top_left, width, height] in pixels
    YOLO bbox format: [class_id, x_center, y_center, width, height] normalized to 0-1

    Each image gets its own .txt file with one line per detection.
    """
    # Load COCO JSON
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    print(f"\n  Converting: {coco_json_path}")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories: {[c['name'] for c in coco['categories']]}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build lookup: image_id → image info
    image_lookup = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Convert each image's annotations
    files_written = 0
    total_boxes = 0

    for img_id, image_info in image_lookup.items():
        img_width = image_info["width"]
        img_height = image_info["height"]
        img_name = Path(image_info["file_name"]).stem  # remove extension

        # Get annotations for this image
        annotations = annotations_by_image.get(img_id, [])

        # Build YOLO format lines
        yolo_lines = []
        for ann in annotations:
            # COCO bbox: [x, y, width, height] in pixels
            x, y, w, h = ann["bbox"]

            # Convert to YOLO format: [x_center, y_center, width, height] normalized
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # Category ID (YOLO uses 0-indexed, so subtract 1)
            class_id = ann["category_id"] - 1

            # Clamp values to [0, 1] to avoid out-of-bounds
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            total_boxes += 1

        # Write YOLO label file (even if empty — YOLO needs empty files for negative samples)
        label_path = os.path.join(output_dir, f"{img_name}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
        files_written += 1

    # Also create a classes.txt file (needed for YOLO training)
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w") as f:
        for cat in coco["categories"]:
            f.write(f"{cat['name']}\n")

    print(f"\n  ✓ Conversion complete!")
    print(f"    Label files written: {files_written}")
    print(f"    Total bounding boxes: {total_boxes}")
    print(f"    Classes file: {classes_path}")
    print(f"    Output directory: {output_dir}")

    return files_written, total_boxes


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO JSON annotations to YOLO format",
        epilog="""
Examples:
  python coco_to_yolo.py --coco data/annotations/video_annotations.json --output data/yolo_labels/
  python coco_to_yolo.py --coco-dir data/annotations/ --output data/yolo_labels/
        """,
    )
    parser.add_argument("--coco", type=str, help="Path to a single COCO JSON file")
    parser.add_argument("--coco-dir", type=str, help="Directory of COCO JSON files")
    parser.add_argument("--output", type=str, default="data/yolo_labels", help="Output directory for YOLO labels")

    args = parser.parse_args()

    if not args.coco and not args.coco_dir:
        parser.print_help()
        sys.exit(1)

    if args.coco:
        convert_coco_to_yolo(args.coco, args.output)

    elif args.coco_dir:
        coco_dir = Path(args.coco_dir)
        json_files = sorted(coco_dir.glob("*.json"))

        if not json_files:
            print(f"  No JSON files found in {coco_dir}")
            sys.exit(1)

        print(f"  Found {len(json_files)} COCO JSON files")
        total_files = 0
        total_boxes = 0

        for jf in json_files:
            f, b = convert_coco_to_yolo(str(jf), args.output)
            total_files += f
            total_boxes += b

        print(f"\n{'='*60}")
        print(f"  All done! {total_files} label files, {total_boxes} boxes total")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
