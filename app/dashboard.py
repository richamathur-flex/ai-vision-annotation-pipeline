"""
Streamlit Dashboard — Detection Statistics & Pipeline Metrics
==============================================================
Visual dashboard showing:
- Model performance metrics
- Detection statistics across all processed videos
- Vehicle type distribution
- Training history
- Pipeline architecture overview

Usage:
    streamlit run dashboard.py

Author: Richa Mathur
"""

import json
import sys
from pathlib import Path

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    print("Run: pip install streamlit pandas plotly")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Vision Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data
def load_all_detections():
    """Load all COCO JSON annotations and aggregate stats."""
    all_data = []
    
    for json_file in ANNOTATIONS_DIR.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        video_name = json_file.stem.replace("_annotations", "")
        
        # Count detections by category
        cat_lookup = {c["id"]: c["name"] for c in data.get("categories", [])}
        
        for ann in data.get("annotations", []):
            cat_id = ann.get("category_id")
            cat_name = cat_lookup.get(cat_id, "unknown")
            all_data.append({
                "video": video_name,
                "frame_id": ann.get("image_id"),
                "class": cat_name,
                "confidence": ann.get("confidence", 0),
                "area": ann.get("area", 0),
            })
    
    return pd.DataFrame(all_data)


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════

st.title("📊 AI Vision Pipeline Dashboard")
st.caption("End-to-end toll road vehicle detection • Richa Mathur")

# Load data
df = load_all_detections()

if df.empty:
    st.warning("No annotation data found. Run auto_annotate.py first.")
    st.stop()

# ─── KPI METRICS ──────────────────────────────────────────────────

st.markdown("### 🎯 Pipeline Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Videos Processed",
        df["video"].nunique(),
        help="Unique videos analyzed"
    )

with col2:
    st.metric(
        "Total Detections",
        len(df),
        help="Total vehicles detected across all videos"
    )

with col3:
    st.metric(
        "Vehicle Types",
        df["class"].nunique(),
        help="Unique vehicle classes"
    )

with col4:
    avg_conf = df["confidence"].mean() if len(df) > 0 else 0
    st.metric(
        "Avg Confidence",
        f"{avg_conf:.1%}",
        help="Average YOLO confidence score"
    )

st.markdown("---")

# ─── MODEL PERFORMANCE ────────────────────────────────────────────

st.markdown("### 🏆 Model Performance (Validation Set)")

perf_data = {
    "Class": ["All", "Car", "Bus", "Truck"],
    "mAP50": [0.806, 0.977, 0.962, 0.481],
    "Precision": [0.725, 0.914, 0.762, 0.500],
    "Recall": [0.857, 0.972, 0.900, 0.700],
}
perf_df = pd.DataFrame(perf_data)

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="mAP50",
        x=perf_df["Class"],
        y=perf_df["mAP50"],
        marker_color="#2a7d4f",
    ))
    fig.add_trace(go.Bar(
        name="Precision",
        x=perf_df["Class"],
        y=perf_df["Precision"],
        marker_color="#2a5fa5",
    ))
    fig.add_trace(go.Bar(
        name="Recall",
        x=perf_df["Class"],
        y=perf_df["Recall"],
        marker_color="#c9a84c",
    ))
    fig.update_layout(
        title="YOLOv8 Fine-Tuned Performance",
        yaxis_title="Score",
        barmode="group",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.caption("Trained for 30 epochs on 118 images")

st.markdown("---")

# ─── VEHICLE TYPE DISTRIBUTION ────────────────────────────────────

st.markdown("### 🚗 Vehicle Type Distribution")

col1, col2 = st.columns(2)

with col1:
    class_counts = df["class"].value_counts().reset_index()
    class_counts.columns = ["class", "count"]
    
    fig = px.pie(
        class_counts,
        values="count",
        names="class",
        title="Detections by Vehicle Type",
        color_discrete_sequence=["#2a7d4f", "#2a5fa5", "#c9a84c", "#a63d2f"],
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        class_counts,
        x="class",
        y="count",
        title="Vehicle Count by Type",
        color="class",
        color_discrete_sequence=["#2a7d4f", "#2a5fa5", "#c9a84c", "#a63d2f"],
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─── VIDEO-LEVEL ANALYSIS ─────────────────────────────────────────

st.markdown("### 📹 Per-Video Analysis")

video_stats = df.groupby("video").agg(
    total_detections=("class", "count"),
    avg_confidence=("confidence", "mean"),
    unique_classes=("class", "nunique"),
).reset_index()

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        video_stats.sort_values("total_detections", ascending=True),
        x="total_detections",
        y="video",
        title="Detections per Video",
        orientation="h",
        color="total_detections",
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        video_stats,
        x="total_detections",
        y="avg_confidence",
        size="unique_classes",
        hover_name="video",
        title="Video Quality: Detections vs Confidence",
        color="avg_confidence",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─── CONFIDENCE DISTRIBUTION ──────────────────────────────────────

st.markdown("### 📈 Confidence Score Distribution")

fig = px.histogram(
    df,
    x="confidence",
    nbins=30,
    title="Distribution of Detection Confidence Scores",
    color="class",
    color_discrete_sequence=["#2a7d4f", "#2a5fa5", "#c9a84c", "#a63d2f"],
)
fig.update_layout(
    xaxis_title="Confidence Score",
    yaxis_title="Number of Detections",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# ─── PIPELINE ARCHITECTURE ────────────────────────────────────────

st.markdown("---")
st.markdown("### 🏗️ Pipeline Architecture")

st.code("""
Raw Videos (H.264/AVI) → ffmpeg → MP4
         ↓
Auto-Annotation (YOLOv8 pre-trained) → COCO JSON
         ↓
Manual Review in CVAT (optional) → Verified annotations
         ↓
Format Conversion → YOLO labels (normalized coordinates)
         ↓
Fine-Tuning (30 epochs) → Custom model (best.pt) [80.6% mAP]
         ↓
FastAPI Deployment → REST endpoint at :8000/docs
         ↓
LangGraph Orchestration
  ├── YOLO Detection (fast)
  ├── Confidence Check (routing)
  ├── GPT-4o Vision (fallback for edge cases)
  └── Claude Analysis (natural language reports)
         ↓
Streamlit Review App → Interactive video review + this dashboard
""", language="text")

# ─── FOOTER ───────────────────────────────────────────────────────

st.markdown("---")
st.caption("""
**Tech Stack:** Python, YOLOv8, OpenCV, FastAPI, Anthropic Claude, OpenAI GPT-4o,
LangGraph, Streamlit, Plotly, Docker  
**Built by:** Richa Mathur • [GitHub](https://github.com/richamathur-flex/ai-vision-annotation-pipeline) •
[LinkedIn](https://linkedin.com/in/richamathurr)
""")
