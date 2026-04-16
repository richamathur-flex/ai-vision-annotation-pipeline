"""
GPT-4o Vision Fallback — Second Opinion on Uncertain Frames
=============================================================
When YOLO confidence is low, send the frame image to GPT-4o
for a second opinion. Demonstrates multi-model AI orchestration.

What this script does:
1. Identifies low-confidence detections from YOLO
2. Sends the frame image to GPT-4o Vision API
3. GPT-4o describes what it sees in natural language
4. Returns analysis to be combined with YOLO results

Why this matters:
- YOLO is fast but can be wrong on edge cases (occlusion, weird angles)
- GPT-4o is slower but more flexible (can reason about context)
- Combining them = best of both worlds
- This is "human-in-the-loop" automated

Usage:
    python gpt_vision_fallback.py --frame ../data/sample_frames/sample.png
    python gpt_vision_fallback.py --frame frame.png --question "How many trucks?"

Author: Richa Mathur
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Run: pip install openai")
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

GPT_MODEL = "gpt-4o"
MAX_TOKENS = 800

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("\n  ⚠ OPENAI_API_KEY not found in .env file!")
    print("  Add to .env: OPENAI_API_KEY=sk-your-key-here")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# GPT-4o VISION ANALYZER
# ═══════════════════════════════════════════════════════════════════

class GPTVisionAnalyzer:
    """
    Uses GPT-4o Vision to analyze frames YOLO is uncertain about.
    """
    
    def __init__(self, api_key=OPENAI_API_KEY, model=GPT_MODEL):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    @staticmethod
    def encode_image(image_path):
        """Convert image to base64 for API transmission."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def analyze_frame(self, image_path, yolo_context=None, custom_question=None):
        """
        Send a frame to GPT-4o for analysis.
        
        Parameters:
            image_path: Path to the frame image (PNG/JPG)
            yolo_context: What YOLO detected (for context)
            custom_question: Specific question about the frame
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Build the prompt
        if custom_question:
            prompt = custom_question
        else:
            prompt = """You are reviewing a traffic surveillance frame from a toll road camera.

Please analyze this image and provide:

1. **Vehicle Count**: How many vehicles do you see total?
2. **Vehicle Types**: List each vehicle (car, truck, bus, motorcycle, trailer)
3. **Axle Counts**: For trucks/buses, estimate axle count (important for toll calculation)
4. **Notable Details**: Any occluded vehicles, unusual angles, or detection challenges
5. **Confidence Assessment**: Rate the image quality (clear/moderate/difficult)

Format as JSON:
{
  "total_vehicles": int,
  "vehicles": [{"type": str, "axles": int, "lane": str, "notes": str}],
  "image_quality": str,
  "challenges": str
}"""
        
        if yolo_context:
            prompt += f"\n\nFor reference, YOLO detected: {json.dumps(yolo_context)}"
            prompt += "\nPlease verify or correct YOLO's findings."
        
        # Encode image
        base64_image = self.encode_image(image_path)
        
        print(f"\n  🔍 Sending frame to GPT-4o Vision...")
        print(f"     Image: {image_path.name}")
        
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )
        
        analysis_text = response.choices[0].message.content
        
        # Try to parse as JSON if it looks like JSON
        analysis_data = None
        if "{" in analysis_text:
            try:
                # Extract JSON from response
                start = analysis_text.find("{")
                end = analysis_text.rfind("}") + 1
                json_str = analysis_text[start:end]
                analysis_data = json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return {
            "image": str(image_path),
            "raw_analysis": analysis_text,
            "structured": analysis_data,
            "model": self.model,
            "tokens": {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            },
        }
    
    def review_low_confidence_detections(self, detection_results, frames_dir, threshold=0.5):
        """
        Find all low-confidence detections and send those frames to GPT-4o.
        """
        frames_dir = Path(frames_dir)
        low_confidence_frames = []
        
        # Find frames with low-confidence detections
        if "detections" in detection_results:
            frames = detection_results["detections"]
        elif "frames" in detection_results:
            frames = detection_results["frames"]
        else:
            return []
        
        for frame in frames:
            vehicles = frame.get("vehicles") or frame.get("detections", [])
            low_conf_vehicles = [v for v in vehicles if v.get("confidence", 1) < threshold]
            
            if low_conf_vehicles:
                low_confidence_frames.append({
                    "frame": frame.get("frame") or frame.get("frame_number"),
                    "low_conf_count": len(low_conf_vehicles),
                    "yolo_detections": low_conf_vehicles,
                })
        
        print(f"\n  Found {len(low_confidence_frames)} frames with low-confidence detections")
        
        # Analyze each (limit to first 3 to save API costs)
        gpt_reviews = []
        for frame_info in low_confidence_frames[:3]:
            print(f"\n  Analyzing frame {frame_info['frame']}...")
            # In real use, you'd construct the actual frame path
            # For now, this is a placeholder showing the pattern
            gpt_reviews.append({
                "frame": frame_info["frame"],
                "yolo_said": frame_info["yolo_detections"],
                "gpt_review": "Would analyze frame here in production",
            })
        
        return gpt_reviews


def main():
    parser = argparse.ArgumentParser(description="GPT-4o Vision analysis of traffic frames")
    parser.add_argument("--frame", required=True, help="Path to frame image")
    parser.add_argument("--question", help="Custom question about the frame")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  GPT-4o Vision Fallback Analyzer")
    print(f"  Model: {GPT_MODEL}")
    print(f"{'='*60}")
    
    analyzer = GPTVisionAnalyzer()
    result = analyzer.analyze_frame(args.frame, custom_question=args.question)
    
    print(f"\n  ✓ Analysis complete")
    print(f"     Tokens used: {result['tokens']['input']} input + {result['tokens']['output']} output")
    print(f"\n  GPT-4o said:")
    print(f"  {'-'*58}")
    print(result["raw_analysis"])
    print(f"  {'-'*58}")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  ✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
