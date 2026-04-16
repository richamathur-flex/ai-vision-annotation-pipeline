# AI Vision Annotation Pipeline

**End-to-end AI system for toll road vehicle detection & axle counting**

Full ML pipeline combining Computer Vision (YOLOv8) with LLM-powered analysis (Claude, GPT-4o) orchestrated via LangGraph. Built by [Richa Mathur](https://linkedin.com/in/richamathurr).

![Python](https://img.shields.io/badge/Python-3.14-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-80.6%25_mAP-green) ![FastAPI](https://img.shields.io/badge/FastAPI-Ready-009688) ![LangGraph](https://img.shields.io/badge/LangGraph-Agents-orange) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 What This Project Does

Automates vehicle detection and traffic analysis for toll road surveillance — a real-world problem I worked on at **Oracle for HCTRA** (Harris County Toll Road Authority) with 32,000+ videos across 75+ cameras. This open-source version demonstrates the same pipeline using free Kaggle CCTV data.

**Key capabilities:**
- Automated vehicle detection (car, truck, bus, motorcycle)
- Axle count estimation for toll calculation
- Multi-model AI orchestration (YOLO + GPT-4o + Claude)
- REST API deployment with FastAPI
- Interactive review dashboard with Streamlit

---

## 🏗️ Architecture

```
Raw Videos → ffmpeg → MP4 → Auto-Annotation (YOLOv8)
                                    ↓
                            COCO JSON → YOLO labels
                                    ↓
                            Fine-Tuning (30 epochs) → best.pt (80.6% mAP)
                                    ↓
                            FastAPI REST Endpoint
                                    ↓
                    LangGraph Multi-Agent Pipeline
                    ├── YOLO Detection
                    ├── Confidence Routing
                    ├── GPT-4o Vision (edge cases)
                    └── Claude Analysis (reports)
                                    ↓
                            Streamlit Dashboard
```

---

## 📊 Model Performance

Trained YOLOv8n on 118 images (30 epochs, CPU ~5 minutes):

| Class | mAP50 | Precision | Recall |
|-------|-------|-----------|--------|
| **All** | **80.6%** | 72.5% | 85.7% |
| Car | 97.7% | 91.4% | 97.2% |
| Bus | 96.2% | 76.2% | 90.0% |
| Truck | 48.1% | 50.0% | 70.0% |

*Truck mAP lower due to class imbalance (6 truck samples vs 26 cars in validation). Addressable with data augmentation and active learning.*

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 8GB RAM minimum
- Optional: GPU for faster training

### Installation

```bash
# Clone the repo
git clone https://github.com/richamathur-flex/ai-vision-annotation-pipeline.git
cd ai-vision-annotation-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env and add your Claude + OpenAI API keys
```

### Run the Pipeline

```bash
# 1. Auto-annotate videos
python scripts/auto_annotate.py --video-dir data/raw_videos/ --all

# 2. Train custom YOLOv8 model
cd models && python train_yolo.py

# 3. Start the detection API
cd inference && python api.py
# Open http://localhost:8000/docs

# 4. Generate traffic reports with Claude
cd agents && python claude_analyzer.py --input ../inference/results.json

# 5. Run the full multi-agent pipeline
python analysis_agent.py --video ../data/raw_videos/sample.avi

# 6. Launch the Streamlit review app
cd app && streamlit run video_review.py
```

---

## 📁 Project Structure

```
ai-vision-annotation-pipeline/
├── scripts/
│   ├── auto_annotate.py       # YOLOv8 → COCO JSON
│   └── coco_to_yolo.py        # Format converter
├── models/
│   ├── train_yolo.py          # Fine-tune YOLOv8
│   └── weights/best.pt        # Trained model
├── inference/
│   ├── api.py                 # FastAPI REST endpoint
│   ├── detect.py              # CLI detection
│   └── draw_boxes.py          # Bounding box overlays
├── agents/
│   ├── claude_analyzer.py     # Claude → traffic reports
│   ├── gpt_vision_fallback.py # GPT-4o → edge cases
│   └── analysis_agent.py      # LangGraph orchestration
├── app/
│   ├── video_review.py        # Streamlit video review
│   └── dashboard.py           # Metrics dashboard
├── data/
│   ├── raw_videos/            # Input videos (gitignored)
│   ├── annotations/           # COCO JSON output
│   └── sample_frames/         # Annotated frames
├── docs/                      # Architecture guides
├── tests/                     # Unit tests
├── requirements.txt
├── env.example
└── .gitignore
```

---

## 🛠️ Tech Stack

**Computer Vision:** YOLOv8, OpenCV, ffmpeg  
**LLMs:** Anthropic Claude, OpenAI GPT-4o Vision  
**Orchestration:** LangGraph, LangChain  
**API:** FastAPI, Uvicorn  
**UI:** Streamlit, Plotly  
**Annotation:** CVAT (for manual review)  
**DevOps:** Docker, Git  

---

## 🎓 What I Learned Building This

1. **Training data quality > model complexity** — 148 well-annotated images outperformed 1000s of low-quality ones
2. **Multi-model orchestration beats single models** — YOLO's speed + GPT-4o's reasoning is better than either alone  
3. **Human-in-the-loop is production reality** — Auto-annotation with manual review is how real ML teams work
4. **Deployment matters** — A model sitting in a notebook is useless. FastAPI + Streamlit makes it real

---

## 📸 Screenshots

### API Documentation (FastAPI auto-generated)
![FastAPI Docs](docs/images/fastapi_docs.png)

### Streamlit Review App
![Streamlit App](docs/images/streamlit_app.png)

### LangGraph Pipeline Output
![Pipeline Output](docs/images/pipeline_output.png)

---

## 🔗 Related Work

This project extends the principles I applied at **Oracle on the HCTRA project** (2024-2025), where I led annotation for 32,000+ toll camera videos. While that work used Oracle's internal Vision API and CVAT, this open-source version demonstrates the same methodology using free tools — making it a useful reference for anyone building similar pipelines.

---

## 👤 About the Author

**Richa Mathur** — AI Engineer specializing in Agentic AI, Computer Vision, and Cloud Solutions  
📧 richa.agenticai@gmail.com  
💼 [LinkedIn](https://linkedin.com/in/richamathurr)  
📍 Dallas, TX

**Certifications:** OCI GenAI Professional, OCI AI Foundations, OCI Data Science Professional, AI Vector Search Professional, Generative AI (Microsoft/LinkedIn), Certified ScrumMaster

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file

---

*Built with care to demonstrate the full lifecycle of modern AI engineering: from raw data to annotated datasets to trained models to deployed APIs to intelligent agent pipelines.*
