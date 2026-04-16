"""
Streamlit Video Review App — Toll Road Vehicle Detection
==========================================================
Interactive web app for reviewing detected vehicles in videos.
Mimics Ashutosh's HCTRA video review tool from the Oracle project.

Features:
- Video selection (dropdown by camera)
- Playback controls (play/pause, speed)
- Detection overlays (show/hide bounding boxes)
- Confidence threshold slider
- Vehicle count per frame
- Detection stats summary

Usage:
    streamlit run video_review.py

Then open browser to: http://localhost:8501

Author: Richa Mathur
"""

import json
import os
import sys
import tempfile
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("Run: pip install streamlit")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("Run: pip install opencv-python")
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
VIDEOS_DIR = PROJECT_ROOT / "data" / "raw_videos"
TRAINED_MODEL = PROJECT_ROOT / "models" / "weights" / "best.pt"

VEHICLE_CLASSES_CUSTOM = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
VEHICLE_CLASSES_PRETRAINED = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

COLORS = {
    "car": (0, 200, 0),
    "motorcycle": (255, 165, 0),
    "bus": (200, 0, 0),
    "truck": (0, 0, 200),
}


# ═══════════════════════════════════════════════════════════════════
# CACHED MODEL LOADER
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load YOLO model once and cache it."""
    if TRAINED_MODEL.exists():
        model = YOLO(str(TRAINED_MODEL))
        return model, VEHICLE_CLASSES_CUSTOM, "fine-tuned"
    else:
        model = YOLO("yolov8n.pt")
        return model, VEHICLE_CLASSES_PRETRAINED, "pre-trained"


# ═══════════════════════════════════════════════════════════════════
# DETECTION & DRAWING
# ═══════════════════════════════════════════════════════════════════

def process_frame_with_detection(frame, model, class_map, confidence, is_custom):
    """Run YOLO on a frame and draw bounding boxes."""
    results = model(frame, conf=confidence, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            
            if not is_custom and cls_id not in VEHICLE_CLASSES_PRETRAINED:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            conf = float(boxes.conf[i])
            class_name = class_map.get(cls_id, "vehicle")
            color = COLORS.get(class_name, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })
    
    return frame, detections


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Toll Road Vehicle Detection",
        page_icon="🚗",
        layout="wide",
    )
    
    # ── Header ──
    st.title("🚗 Toll Road Vehicle Detection Review")
    st.caption("AI-powered vehicle detection and review • Built by Richa Mathur")
    
    # ── Load Model ──
    with st.spinner("Loading AI model..."):
        model, class_map, model_type = load_model()
    
    # ── Sidebar Controls ──
    with st.sidebar:
        st.header("⚙️ Controls")
        
        st.success(f"✓ Model loaded: **{model_type}**")
        
        # Video selection
        st.subheader("📹 Video Selection")
        video_files = sorted(list(VIDEOS_DIR.glob("*.avi")) + list(VIDEOS_DIR.glob("*.mp4")))
        
        if not video_files:
            st.error(f"No videos found in {VIDEOS_DIR}")
            st.stop()
        
        video_names = [v.name for v in video_files]
        selected_video = st.selectbox(
            "Choose video:",
            video_names,
            help="Select a toll road camera video to analyze"
        )
        
        # Detection controls
        st.subheader("🎯 Detection")
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.15,
            step=0.05,
            help="Lower = more detections, Higher = stricter"
        )
        
        show_detections = st.checkbox("Show bounding boxes", value=True)
        show_stats = st.checkbox("Show statistics", value=True)
        
        # Playback controls
        st.subheader("▶️ Playback")
        frame_interval = st.select_slider(
            "Frame sampling",
            options=[1, 2, 5, 10, 15, 30],
            value=5,
            help="Process every Nth frame"
        )
        
        st.markdown("---")
        st.caption("Built with Streamlit + YOLOv8")
    
    # ── Main Content ──
    video_path = VIDEOS_DIR / selected_video
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"📺 {selected_video}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            st.error(f"Could not open video: {selected_video}")
            st.stop()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Frame selector
        frame_num = st.slider(
            "Select frame",
            min_value=0,
            max_value=total_frames - 1,
            value=0,
            step=frame_interval,
        )
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Run detection
            if show_detections:
                frame, detections = process_frame_with_detection(
                    frame, model, class_map, confidence,
                    is_custom=(model_type == "fine-tuned")
                )
            else:
                # Still detect to show stats, but don't draw
                detections_only = []
                results = model(frame, conf=confidence, verbose=False)
                for r in results:
                    if r.boxes is not None:
                        for i in range(len(r.boxes)):
                            cls_id = int(r.boxes.cls[i])
                            class_name = class_map.get(cls_id, "vehicle")
                            detections_only.append({
                                "class": class_name,
                                "confidence": float(r.boxes.conf[i]),
                            })
                detections = detections_only
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)
            
            st.caption(f"Frame {frame_num}/{total_frames} • Time: {frame_num/fps:.1f}s • {len(detections)} vehicles detected")
    
    with col2:
        if show_stats:
            st.subheader("📊 Statistics")
            
            # Video info
            with st.container(border=True):
                st.markdown("**Video Info**")
                st.metric("Total frames", total_frames)
                st.metric("Resolution", f"{width}×{height}")
                st.metric("FPS", f"{fps:.1f}")
            
            # Current frame detections
            with st.container(border=True):
                st.markdown("**This Frame**")
                if detections:
                    st.metric("Vehicles detected", len(detections))
                    
                    # Class breakdown
                    class_counts = {}
                    for d in detections:
                        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1
                    
                    st.markdown("**Breakdown:**")
                    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                        st.text(f"  {cls}: {count}")
                    
                    # Avg confidence
                    avg_conf = sum(d["confidence"] for d in detections) / len(detections)
                    st.metric("Avg confidence", f"{avg_conf:.0%}")
                else:
                    st.info("No vehicles detected in this frame")
            
            # Action buttons
            with st.container(border=True):
                st.markdown("**Actions**")
                if st.button("📥 Export detections as JSON", use_container_width=True):
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(detections, indent=2),
                        file_name=f"{selected_video}_frame_{frame_num}.json",
                        mime="application/json",
                    )
    
    # ── Bottom: Full Video Analysis ──
    st.markdown("---")
    
    if st.button("🔍 Analyze Full Video", type="primary"):
        analyze_full_video(video_path, model, class_map, confidence,
                          model_type == "fine-tuned", frame_interval)


def analyze_full_video(video_path, model, class_map, confidence, is_custom, interval):
    """Run detection on all frames and show summary."""
    st.subheader("📹 Full Video Analysis")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress = st.progress(0)
    status = st.empty()
    
    frame_count = 0
    total_detections = 0
    class_totals = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % interval != 0:
            continue
        
        results = model(frame, conf=confidence, verbose=False)
        for r in results:
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i])
                    if not is_custom and cls_id not in VEHICLE_CLASSES_PRETRAINED:
                        continue
                    class_name = class_map.get(cls_id, "vehicle")
                    class_totals[class_name] = class_totals.get(class_name, 0) + 1
                    total_detections += 1
        
        progress.progress(frame_count / total_frames)
        status.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    progress.empty()
    status.empty()
    
    # Display results
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total vehicles", total_detections)
    with c2:
        frames_processed = frame_count // interval
        avg = total_detections / max(frames_processed, 1)
        st.metric("Avg per frame", f"{avg:.1f}")
    with c3:
        st.metric("Frames analyzed", frames_processed)
    with c4:
        st.metric("Vehicle types", len(class_totals))
    
    st.markdown("**Breakdown by type:**")
    for cls, count in sorted(class_totals.items(), key=lambda x: -x[1]):
        st.text(f"  • {cls}: {count} ({100*count/max(total_detections,1):.1f}%)")


if __name__ == "__main__":
    main()
