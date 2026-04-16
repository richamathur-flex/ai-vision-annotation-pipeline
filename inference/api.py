"""
FastAPI Inference API — Vehicle Detection Endpoint
====================================================
Serves the trained YOLOv8 model as a REST API.

Endpoints:
    GET  /health          → Check if API is running
    POST /detect          → Upload video/image, get detection JSON
    POST /detect/frame    → Upload single image frame
    GET  /stats           → Model info and detection stats

Usage:
    # Start the server:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

    # Test with curl:
    curl -X POST http://localhost:8000/detect/frame -F "file=@frame.png"

    # Or open browser: http://localhost:8000/docs (auto-generated docs)

Why FastAPI?
    - Auto-generates API documentation (Swagger UI)
    - Type-safe with Python type hints
    - Async support for handling multiple requests
    - Industry standard for ML model serving
    - Shows recruiters you can DEPLOY models, not just train them

Author: Richa Mathur
"""

import io
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("Run: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Run: pip install ultralytics")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent

# Try trained model first, fall back to pre-trained
TRAINED_MODEL = PROJECT_ROOT / "models" / "weights" / "best.pt"
PRETRAINED_MODEL = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.3

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# If using fine-tuned model, classes are 0-indexed
CUSTOM_CLASSES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}

# Track stats
detection_stats = {"total_requests": 0, "total_detections": 0, "start_time": None}


# ═══════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════

def load_model():
    """Load the best available model."""
    if TRAINED_MODEL.exists():
        print(f"  ✓ Loading fine-tuned model: {TRAINED_MODEL}")
        model = YOLO(str(TRAINED_MODEL))
        using_custom = True
    else:
        print(f"  ⚠ Fine-tuned model not found. Using pre-trained: {PRETRAINED_MODEL}")
        print(f"    Run models/train_yolo.py first for better results.")
        model = YOLO(PRETRAINED_MODEL)
        using_custom = False
    return model, using_custom


model, using_custom_classes = load_model()
class_map = CUSTOM_CLASSES if using_custom_classes else VEHICLE_CLASSES


# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Toll Road Vehicle Detection API",
    description="YOLOv8-powered vehicle detection and axle counting for toll road surveillance",
    version="1.0.0",
    contact={"name": "Richa Mathur", "email": "richa.agenticai@gmail.com"},
)


@app.on_event("startup")
async def startup():
    detection_stats["start_time"] = datetime.now().isoformat()
    print("\n  ═══════════════════════════════════════")
    print("  Toll Road Vehicle Detection API")
    print("  Open docs: http://localhost:8000/docs")
    print("  ═══════════════════════════════════════\n")


# ── Health Check ──────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Check if the API is running and model is loaded."""
    return {
        "status": "healthy",
        "model": str(TRAINED_MODEL) if TRAINED_MODEL.exists() else PRETRAINED_MODEL,
        "model_type": "fine-tuned" if using_custom_classes else "pre-trained",
        "classes": list(class_map.values()),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


# ── Detect from single image/frame ───────────────────────────────

@app.post("/detect/frame")
async def detect_frame(
    file: UploadFile = File(...),
    confidence: float = CONFIDENCE_THRESHOLD,
):
    """
    Detect vehicles in a single image frame.
    
    Upload a PNG/JPG image and get back:
    - List of detected vehicles with bounding boxes
    - Confidence scores
    - Vehicle class (car, truck, bus, motorcycle)
    - Placeholder attributes for manual review (axle count, direction, lane)
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (PNG, JPG)")
    
    start_time = time.time()
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(400, "Could not decode image")
    
    h, w = frame.shape[:2]
    
    # Run YOLO
    results = model(frame, conf=confidence, verbose=False)
    
    # Build detections list
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            
            # Filter to vehicles only (if using pre-trained model)
            if not using_custom_classes and class_id not in VEHICLE_CLASSES:
                continue
            
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            
            class_name = class_map.get(class_id, f"class_{class_id}")
            
            detections.append({
                "id": len(detections) + 1,
                "class": class_name,
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                    "width": round(x2 - x1, 1),
                    "height": round(y2 - y1, 1),
                },
                # Attributes that need human review
                "attributes": {
                    "axle_count": None,
                    "direction": None,
                    "lane": None,
                    "is_occluded": None,
                    "needs_review": conf < 0.6,
                },
            })
    
    inference_time = round((time.time() - start_time) * 1000, 1)
    
    # Update stats
    detection_stats["total_requests"] += 1
    detection_stats["total_detections"] += len(detections)
    
    return {
        "success": True,
        "filename": file.filename,
        "image_size": {"width": w, "height": h},
        "inference_time_ms": inference_time,
        "total_detections": len(detections),
        "detections": detections,
        "low_confidence_count": sum(1 for d in detections if d["confidence"] < 0.6),
        "model_type": "fine-tuned" if using_custom_classes else "pre-trained",
    }


# ── Detect from video file ───────────────────────────────────────

@app.post("/detect")
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = CONFIDENCE_THRESHOLD,
    frame_interval: int = 5,
):
    """
    Detect vehicles in a video file.
    
    Processes every Nth frame (default: every 5th).
    Returns frame-by-frame detections.
    """
    start_time = time.time()
    
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_results = []
        frame_count = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue
            
            # Run YOLO
            results = model(frame, conf=confidence, verbose=False)
            
            frame_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    if not using_custom_classes and class_id not in VEHICLE_CLASSES:
                        continue
                    
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    conf = float(boxes.conf[i])
                    class_name = class_map.get(class_id, f"class_{class_id}")
                    
                    frame_detections.append({
                        "class": class_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    })
            
            total_detections += len(frame_detections)
            frame_results.append({
                "frame_number": frame_count,
                "timestamp_sec": round(frame_count / max(fps, 1), 2),
                "detections": frame_detections,
                "vehicle_count": len(frame_detections),
            })
        
        cap.release()
        
    finally:
        os.unlink(tmp_path)
    
    inference_time = round((time.time() - start_time) * 1000, 1)
    
    detection_stats["total_requests"] += 1
    detection_stats["total_detections"] += total_detections
    
    return {
        "success": True,
        "filename": file.filename,
        "video_info": {
            "width": width,
            "height": height,
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "frames_processed": len(frame_results),
        },
        "inference_time_ms": inference_time,
        "total_detections": total_detections,
        "avg_vehicles_per_frame": round(total_detections / max(len(frame_results), 1), 1),
        "frames": frame_results,
    }


# ── Stats ─────────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    return {
        "api_start_time": detection_stats["start_time"],
        "total_requests": detection_stats["total_requests"],
        "total_detections": detection_stats["total_detections"],
        "model": str(TRAINED_MODEL) if TRAINED_MODEL.exists() else PRETRAINED_MODEL,
        "classes": list(class_map.values()),
    }


# ═══════════════════════════════════════════════════════════════════
# RUN SERVER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
