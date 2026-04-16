"""
Train YOLOv8 on Toll Road Vehicle Detection Data
==================================================
This script fine-tunes a pre-trained YOLOv8 model on your
auto-annotated traffic surveillance data.

What this script does:
1. Reads your COCO JSON annotations (from auto_annotate.py)
2. Converts them to YOLO format (using coco_to_yolo.py)
3. Splits data into train/val sets (80/20)
4. Creates a YOLO config.yaml
5. Fine-tunes YOLOv8 on your data
6. Saves the trained model + training metrics

Why fine-tune instead of using pre-trained YOLO?
- Pre-trained YOLO detects 80 classes (people, dogs, chairs...)
- We only need vehicles (car, truck, bus, motorcycle)
- Fine-tuning on toll camera angles improves accuracy
- Shows recruiters you understand transfer learning

Usage:
    python train_yolo.py
    python train_yolo.py --epochs 50 --model yolov8s.pt
    python train_yolo.py --skip-convert  (if YOLO labels already exist)

Author: Richa Mathur
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# ── Add parent directory to path so we can import from scripts/ ──
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from ultralytics import YOLO
except ImportError:
    print("Run: pip install ultralytics")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
FRAMES_DIR = PROJECT_ROOT / "data" / "sample_frames"
YOLO_DATASET_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "weights"

# Class names matching our auto_annotate.py categories
CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]

# Training parameters
DEFAULT_EPOCHS = 30
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_IMG_SIZE = 320        # Match our video resolution (320x240)
DEFAULT_BATCH_SIZE = 8
TRAIN_VAL_SPLIT = 0.8         # 80% train, 20% validation


# ═══════════════════════════════════════════════════════════════════
# STEP 1: CONVERT COCO JSON → YOLO FORMAT
# ═══════════════════════════════════════════════════════════════════

def convert_coco_to_yolo_labels(annotations_dir, frames_dir, output_dir):
    """
    Convert all COCO JSON files to YOLO format.
    
    COCO format: [x, y, width, height] in pixels
    YOLO format: [class_id, x_center, y_center, width, height] normalized 0-1
    
    Also copies the corresponding frame images.
    """
    annotations_dir = Path(annotations_dir)
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    images_dir = output_dir / "images" / "all"
    labels_dir = output_dir / "labels" / "all"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        print(f"  ✗ No JSON files found in {annotations_dir}")
        print(f"    Run auto_annotate.py first to generate annotations.")
        sys.exit(1)
    
    print(f"\n  Converting {len(json_files)} COCO JSON files to YOLO format...")
    
    total_images = 0
    total_labels = 0
    
    for jf in json_files:
        with open(jf) as f:
            coco = json.load(f)
        
        video_name = jf.stem.replace("_annotations", "")
        video_frames_dir = frames_dir / video_name
        
        # Build image lookup
        image_lookup = {img["id"]: img for img in coco["images"]}
        
        # Group annotations by image
        anns_by_image = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)
        
        for img_id, img_info in image_lookup.items():
            img_w = img_info["width"]
            img_h = img_info["height"]
            img_name = img_info["file_name"]
            stem = Path(img_name).stem
            
            # Copy frame image if it exists
            src_img = video_frames_dir / img_name
            if src_img.exists():
                dst_img = images_dir / img_name
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
                total_images += 1
            
            # Convert annotations to YOLO format
            anns = anns_by_image.get(img_id, [])
            yolo_lines = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                # YOLO: class x_center y_center width height (all normalized)
                x_center = min(1, max(0, (x + w / 2) / img_w))
                y_center = min(1, max(0, (y + h / 2) / img_h))
                w_norm = min(1, max(0, w / img_w))
                h_norm = min(1, max(0, h / img_h))
                class_id = ann["category_id"] - 1  # YOLO is 0-indexed
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                total_labels += 1
            
            # Write label file
            label_path = labels_dir / f"{stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
    
    print(f"  ✓ Converted: {total_images} images, {total_labels} labels")
    return images_dir, labels_dir


# ═══════════════════════════════════════════════════════════════════
# STEP 2: SPLIT INTO TRAIN / VALIDATION
# ═══════════════════════════════════════════════════════════════════

def split_train_val(dataset_dir, split_ratio=TRAIN_VAL_SPLIT):
    """
    Split dataset into training and validation sets.
    
    Why split? The model trains on the training set and we check
    accuracy on the validation set. If we test on training data,
    we can't tell if the model actually learned or just memorized.
    """
    all_images = dataset_dir / "images" / "all"
    all_labels = dataset_dir / "labels" / "all"
    
    # Create train/val directories
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(list(all_images.glob("*.png")) + list(all_images.glob("*.jpg")))
    
    if not image_files:
        print(f"  ✗ No images found in {all_images}")
        sys.exit(1)
    
    # Shuffle and split
    random.seed(42)  # Reproducible split
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"\n  Splitting dataset: {len(train_files)} train, {len(val_files)} val")
    
    # Copy files to train/val folders
    for files, split in [(train_files, "train"), (val_files, "val")]:
        for img_path in files:
            # Copy image
            dst_img = dataset_dir / "images" / split / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            
            # Copy matching label
            label_name = img_path.stem + ".txt"
            src_label = all_labels / label_name
            dst_label = dataset_dir / "labels" / split / label_name
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)
    
    print(f"  ✓ Train: {len(train_files)} images | Val: {len(val_files)} images")
    return len(train_files), len(val_files)


# ═══════════════════════════════════════════════════════════════════
# STEP 3: CREATE YOLO CONFIG FILE
# ═══════════════════════════════════════════════════════════════════

def create_yolo_config(dataset_dir):
    """
    Create the data.yaml config file that YOLOv8 needs for training.
    
    This file tells YOLO:
    - Where to find training images
    - Where to find validation images
    - How many classes there are
    - What the class names are
    """
    config_path = dataset_dir / "data.yaml"
    
    config_content = f"""# YOLOv8 Training Configuration
# Auto-generated by train_yolo.py
# Project: AI Vision Annotation Pipeline - Toll Road Vehicle Detection

path: {dataset_dir.resolve()}
train: images/train
val: images/val

# Number of classes
nc: {len(CLASS_NAMES)}

# Class names (must match auto_annotate.py categories)
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        config_content += f"  {i}: {name}\n"
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"\n  ✓ Created config: {config_path}")
    return config_path


# ═══════════════════════════════════════════════════════════════════
# STEP 4: TRAIN THE MODEL
# ═══════════════════════════════════════════════════════════════════

def train_model(config_path, model_name=DEFAULT_MODEL, epochs=DEFAULT_EPOCHS,
                img_size=DEFAULT_IMG_SIZE, batch_size=DEFAULT_BATCH_SIZE):
    """
    Fine-tune YOLOv8 on the toll road dataset.
    
    What happens during training:
    1. Load pre-trained model (already knows general objects)
    2. Replace the last layer for our 4 classes
    3. Show it our annotated frames repeatedly (epochs)
    4. Model adjusts its weights to detect vehicles better
    5. After each epoch, check accuracy on validation set
    6. Save the best model (highest accuracy)
    
    Key metrics to watch:
    - mAP50: Mean Average Precision at 50% IoU (higher = better)
    - Precision: Of all detections, how many were correct?
    - Recall: Of all real vehicles, how many did we find?
    """
    print(f"\n{'='*60}")
    print(f"  Starting YOLOv8 Training")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Load pre-trained model
    model = YOLO(model_name)
    
    # Create output directory
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train!
    results = model.train(
        data=str(config_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=str(PROJECT_ROOT / "models"),
        name="toll_road_detector",
        exist_ok=True,
        
        # Training hyperparameters
        patience=10,          # Stop early if no improvement for 10 epochs
        save=True,            # Save checkpoints
        save_period=5,        # Save every 5 epochs
        plots=True,           # Generate training plots
        verbose=True,
        
        # Data augmentation (helps with small datasets)
        augment=True,
        flipud=0.0,           # Don't flip vertically (cameras are always upright)
        fliplr=0.5,           # Horizontal flip (vehicles can go either direction)
        mosaic=0.5,           # Combine 4 images into 1 (great for small datasets)
    )
    
    # Find best model weights
    best_weights = PROJECT_ROOT / "models" / "toll_road_detector" / "weights" / "best.pt"
    
    if best_weights.exists():
        # Copy to our weights directory
        dst = MODEL_OUTPUT_DIR / "best.pt"
        shutil.copy2(best_weights, dst)
        print(f"\n  ✓ Best model saved to: {dst}")
    
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Results saved to: models/toll_road_detector/")
    print(f"  Training plots: models/toll_road_detector/results.png")
    print(f"  Best weights: models/weights/best.pt")
    print(f"{'='*60}")
    
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on toll road vehicle detection data",
        epilog="""
Examples:
  python train_yolo.py                          # Default training
  python train_yolo.py --epochs 50              # More epochs
  python train_yolo.py --model yolov8s.pt       # Bigger model
  python train_yolo.py --skip-convert           # Skip format conversion
        """,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--skip-convert", action="store_true", help="Skip COCO→YOLO conversion")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  AI Vision Pipeline — YOLOv8 Training")
    print(f"  Toll Road Vehicle Detection")
    print(f"{'='*60}")
    
    # Step 1: Convert annotations
    if not args.skip_convert:
        print(f"\n  STEP 1: Converting COCO → YOLO format")
        convert_coco_to_yolo_labels(ANNOTATIONS_DIR, FRAMES_DIR, YOLO_DATASET_DIR)
    else:
        print(f"\n  STEP 1: Skipping conversion (--skip-convert)")
    
    # Step 2: Split train/val
    print(f"\n  STEP 2: Splitting train/val")
    split_train_val(YOLO_DATASET_DIR)
    
    # Step 3: Create config
    print(f"\n  STEP 3: Creating YOLO config")
    config_path = create_yolo_config(YOLO_DATASET_DIR)
    
    # Step 4: Train
    print(f"\n  STEP 4: Training YOLOv8")
    train_model(config_path, args.model, args.epochs, args.img_size, args.batch)


if __name__ == "__main__":
    main()
