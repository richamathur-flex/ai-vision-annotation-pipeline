"""
Claude Analyzer — Generate Natural Language Reports from Detections
====================================================================
Takes YOLO detection JSON and uses Claude API to generate
human-readable traffic analysis reports.

What this script does:
1. Reads detection results (from detect.py or api.py)
2. Sends to Claude API with structured prompt
3. Claude generates: traffic patterns, anomalies, summary, recommendations
4. Saves report as markdown file

Why use Claude here?
- YOLO outputs raw numbers (bounding boxes, confidence scores)
- Humans need context: "47 cars, peak at 2pm, 3 trucks suspicious"
- Claude turns data into insights
- Demonstrates LLM integration on your resume

Usage:
    python claude_analyzer.py --input results.json
    python claude_analyzer.py --input results.json --output report.md

Author: Richa Mathur
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    print("Run: pip install anthropic")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Run: pip install python-dotenv")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

CLAUDE_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 2000

# Get API key from .env file
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("\n  ⚠ ANTHROPIC_API_KEY not found in .env file!")
    print("  Add to .env: ANTHROPIC_API_KEY=sk-ant-your-key-here")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CLAUDE ANALYZER
# ═══════════════════════════════════════════════════════════════════

class ClaudeTrafficAnalyzer:
    """
    Uses Claude to analyze traffic detection data and generate reports.
    """
    
    def __init__(self, api_key=ANTHROPIC_API_KEY, model=CLAUDE_MODEL):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def summarize_detections(self, detection_data):
        """
        Build a compact summary of detections to send to Claude.
        We don't send raw bounding boxes — too much data, expensive.
        Instead, we send aggregated stats.
        """
        if "detections" in detection_data:
            # Format from detect.py
            frames = detection_data["detections"]
        elif "frames" in detection_data:
            # Format from api.py
            frames = detection_data["frames"]
        else:
            frames = []
        
        # Count by class
        class_counts = {}
        confidence_buckets = {"high (>0.7)": 0, "medium (0.4-0.7)": 0, "low (<0.4)": 0}
        total_vehicles = 0
        max_per_frame = 0
        peak_frame = None
        
        for frame in frames:
            vehicles = frame.get("vehicles") or frame.get("detections", [])
            count = len(vehicles)
            total_vehicles += count
            
            if count > max_per_frame:
                max_per_frame = count
                peak_frame = frame.get("frame") or frame.get("frame_number")
            
            for vehicle in vehicles:
                cls = vehicle.get("class", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1
                
                conf = vehicle.get("confidence", 0)
                if conf > 0.7:
                    confidence_buckets["high (>0.7)"] += 1
                elif conf > 0.4:
                    confidence_buckets["medium (0.4-0.7)"] += 1
                else:
                    confidence_buckets["low (<0.4)"] += 1
        
        return {
            "video": detection_data.get("video", "unknown"),
            "total_frames_processed": len(frames),
            "total_vehicles_detected": total_vehicles,
            "avg_vehicles_per_frame": round(total_vehicles / max(len(frames), 1), 2),
            "vehicle_breakdown": class_counts,
            "confidence_distribution": confidence_buckets,
            "peak_frame": peak_frame,
            "peak_vehicle_count": max_per_frame,
            "model_type": detection_data.get("model", "unknown"),
        }
    
    def generate_report(self, detection_data, context=None):
        """
        Send detection summary to Claude and get back a structured report.
        """
        summary = self.summarize_detections(detection_data)
        
        # Build the prompt
        prompt = f"""You are an AI traffic analyst reviewing toll road surveillance footage.
Analyze the following detection data and generate a professional report.

DETECTION SUMMARY:
{json.dumps(summary, indent=2)}

{f"ADDITIONAL CONTEXT: {context}" if context else ""}

Generate a structured markdown report with these sections:

# Traffic Analysis Report

## Executive Summary
2-3 sentences overview of traffic patterns observed.

## Vehicle Distribution
Breakdown of detected vehicle types with insights.

## Confidence Analysis
Comment on detection reliability and any concerns about low-confidence detections.

## Key Observations
- Peak traffic moments
- Unusual patterns
- Anomalies that need human review

## Recommendations
- Operational recommendations (e.g., "review frame X manually")
- Model improvement suggestions
- Toll calculation considerations (axle counts matter!)

Keep the tone professional and actionable. Use specific numbers from the data.
"""
        
        print("\n  🤖 Asking Claude to analyze detections...")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        
        report = response.content[0].text
        
        # Add metadata footer
        report += f"\n\n---\n*Report generated by Claude {self.model} on {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        report += f"\n*Tokens used: {response.usage.input_tokens} input + {response.usage.output_tokens} output*"
        
        return report, summary


def main():
    parser = argparse.ArgumentParser(description="Generate traffic report from YOLO detections using Claude")
    parser.add_argument("--input", required=True, help="Detection JSON file (from detect.py)")
    parser.add_argument("--output", default=None, help="Output markdown file")
    parser.add_argument("--context", default=None, help="Additional context (e.g., 'Highway I-45')")
    args = parser.parse_args()
    
    # Load detection data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"  ✗ File not found: {input_path}")
        sys.exit(1)
    
    with open(input_path) as f:
        detection_data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"  Claude Traffic Analyzer")
    print(f"  Input: {input_path}")
    print(f"  Model: {CLAUDE_MODEL}")
    print(f"{'='*60}")
    
    # Generate report
    analyzer = ClaudeTrafficAnalyzer()
    report, summary = analyzer.generate_report(detection_data, args.context)
    
    # Print summary
    print(f"\n  📊 Detection Summary:")
    print(f"     Total vehicles: {summary['total_vehicles_detected']}")
    print(f"     Avg per frame: {summary['avg_vehicles_per_frame']}")
    print(f"     Vehicle types: {summary['vehicle_breakdown']}")
    
    # Save or print report
    output_path = args.output or input_path.with_suffix(".report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n  ✓ Report saved: {output_path}")
    print(f"\n{'='*60}")
    print("  REPORT PREVIEW (first 30 lines):")
    print(f"{'='*60}\n")
    print("\n".join(report.split("\n")[:30]))
    print(f"\n  ... (full report in {output_path})")


if __name__ == "__main__":
    main()
