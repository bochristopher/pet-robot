#!/usr/bin/env python3
"""
OpenAI Vision Module (GPT-4V)
Premium vision understanding for the robot pet.

Features:
- Multiple query modes (describe, obstacles, find_person, identify)
- Frame caching to reduce API costs
- Structured JSON responses for robot actions
- Cost tracking
"""

import os
import sys
import cv2
import base64
import time
import json
import hashlib
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# Configuration
CAMERA_INDEX = 0  # /dev/video0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 85

# Cache settings (don't re-analyze same scene within this time)
CACHE_TTL_SECONDS = 3.0
FRAME_SIMILARITY_THRESHOLD = 0.95  # Consider frames "same" if > 95% similar

# Vision prompts for different query types
VISION_PROMPTS = {
    "describe": """Describe what you see in this image in 2-3 sentences.
Focus on the main subjects, their positions, colors, and any notable details.
Be specific but concise - this will be spoken aloud by a robot pet.""",

    "describe_brief": """Describe this scene in one short sentence (under 15 words).
Focus only on the most important element.""",

    "obstacles": """Analyze this image for navigation safety. Respond in JSON format:
{
    "path_clear": true/false,
    "obstacles": ["list of obstacles with positions"],
    "recommended_action": "forward/left/right/stop/backup",
    "confidence": 0.0-1.0
}
Focus on obstacles at ground level that would block a small wheeled robot.""",

    "find_person": """Is there a person visible in this image? Respond in JSON format:
{
    "person_detected": true/false,
    "count": number,
    "positions": ["left", "center", "right"],
    "distance_estimate": "close/medium/far",
    "facing_camera": true/false,
    "description": "brief description"
}""",

    "identify": """Look for {target} in this image. Respond in JSON format:
{{
    "found": true/false,
    "confidence": 0.0-1.0,
    "position": "left/center/right/not_visible",
    "description": "brief description of what you see"
}}
Only report if you're reasonably confident you see {target}.""",

    "mood": """What is the overall mood or atmosphere of this scene?
Describe in 1-2 sentences that capture the feeling.
Consider lighting, colors, activity level, and general vibe.""",

    "explore": """You are a curious robot pet exploring. What's the most interesting thing in this scene?
Describe it enthusiastically in 1-2 sentences, as if you're a curious pet discovering something new.
Be playful and express wonder or curiosity.""",
}


@dataclass
class VisionStats:
    """Track vision API usage."""
    api_calls: int = 0
    cached_hits: int = 0
    total_tokens: int = 0
    errors: int = 0
    last_call_time: float = 0
    
    @property
    def estimated_cost(self) -> float:
        # GPT-4V: ~$0.01 per image (low detail) + tokens
        return self.api_calls * 0.01 + self.total_tokens * 0.00001


stats = VisionStats()


@dataclass
class CachedResult:
    """Cached vision result."""
    result: Any
    timestamp: float
    frame_hash: str


class VisionCache:
    """Simple cache for vision results."""
    
    def __init__(self, ttl: float = CACHE_TTL_SECONDS):
        self.cache: dict[str, CachedResult] = {}
        self.ttl = ttl
    
    def get(self, query_type: str, frame_hash: str) -> Optional[Any]:
        """Get cached result if valid."""
        key = f"{query_type}:{frame_hash}"
        if key in self.cache:
            cached = self.cache[key]
            if time.time() - cached.timestamp < self.ttl:
                return cached.result
            else:
                del self.cache[key]
        return None
    
    def set(self, query_type: str, frame_hash: str, result: Any):
        """Cache a result."""
        key = f"{query_type}:{frame_hash}"
        self.cache[key] = CachedResult(
            result=result,
            timestamp=time.time(),
            frame_hash=frame_hash
        )
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()


class OpenAIVision:
    """
    OpenAI Vision API wrapper for robot pet.
    
    Usage:
        vision = OpenAIVision()
        description = vision.describe_scene()
        obstacles = vision.detect_obstacles()
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        self.camera = None
        self.cache = VisionCache()
        self.last_frame = None
        self.last_frame_hash = None
        
        if not self.api_key:
            print("[Vision] ❌ OPENAI_API_KEY not set!")
            print("         Run: export OPENAI_API_KEY='sk-...'")
        else:
            self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            print(f"[Vision] ✅ OpenAI initialized (key: {self.api_key[:8]}...)")
        except ImportError:
            print("[Vision] ❌ openai library not installed")
            print("         Run: pip install openai")
        except Exception as e:
            print(f"[Vision] ❌ OpenAI init failed: {e}")
    
    def _init_camera(self) -> bool:
        """Initialize camera if needed."""
        if self.camera is not None and self.camera.isOpened():
            return True
        
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            print(f"[Vision] ❌ Cannot open camera {CAMERA_INDEX}")
            return False
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"[Vision] 📷 Camera opened: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        return True
    
    def capture_frame(self) -> Optional[bytes]:
        """Capture a frame from the camera and return as JPEG bytes."""
        if not self._init_camera():
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            print("[Vision] ❌ Failed to capture frame")
            return None
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpeg_bytes = buffer.tobytes()
        
        # Store for reference
        self.last_frame = frame
        self.last_frame_hash = hashlib.md5(jpeg_bytes).hexdigest()[:16]
        
        return jpeg_bytes
    
    def encode_frame_base64(self, jpeg_bytes: bytes) -> str:
        """Encode JPEG bytes to base64."""
        return base64.b64encode(jpeg_bytes).decode('utf-8')
    
    def _call_vision_api(self, prompt: str, jpeg_bytes: bytes, 
                         detail: str = "low", max_tokens: int = 300) -> Optional[str]:
        """Call OpenAI Vision API with an image."""
        global stats
        
        if not self.client:
            return None
        
        try:
            base64_image = self.encode_frame_base64(jpeg_bytes)
            
            stats.api_calls += 1
            stats.last_call_time = time.time()
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # GPT-4V (Vision)
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": detail
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            stats.total_tokens += response.usage.total_tokens if response.usage else 0
            
            return result
            
        except Exception as e:
            stats.errors += 1
            print(f"[Vision] ❌ API error: {e}")
            return None
    
    def describe_scene(self, brief: bool = False) -> Optional[str]:
        """Describe what the camera sees."""
        jpeg_bytes = self.capture_frame()
        if not jpeg_bytes:
            return None
        
        query_type = "describe_brief" if brief else "describe"
        
        # Check cache
        cached = self.cache.get(query_type, self.last_frame_hash)
        if cached:
            stats.cached_hits += 1
            print(f"[Vision] 📀 Cache hit: {query_type}")
            return cached
        
        print(f"[Vision] 🔍 Analyzing scene...")
        prompt = VISION_PROMPTS[query_type]
        result = self._call_vision_api(prompt, jpeg_bytes)
        
        if result:
            self.cache.set(query_type, self.last_frame_hash, result)
        
        return result
    
    def detect_obstacles(self) -> Optional[dict]:
        """Detect obstacles for navigation."""
        jpeg_bytes = self.capture_frame()
        if not jpeg_bytes:
            return None
        
        # Check cache
        cached = self.cache.get("obstacles", self.last_frame_hash)
        if cached:
            stats.cached_hits += 1
            return cached
        
        print(f"[Vision] 🚧 Checking for obstacles...")
        result = self._call_vision_api(VISION_PROMPTS["obstacles"], jpeg_bytes)
        
        if result:
            try:
                # Parse JSON response
                json_str = result
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0]
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0]
                
                parsed = json.loads(json_str.strip())
                self.cache.set("obstacles", self.last_frame_hash, parsed)
                return parsed
            except json.JSONDecodeError:
                print(f"[Vision] ⚠️  Could not parse JSON: {result}")
                return {"raw_response": result, "path_clear": None}
        
        return None
    
    def find_person(self) -> Optional[dict]:
        """Look for a person in the camera view."""
        jpeg_bytes = self.capture_frame()
        if not jpeg_bytes:
            return None
        
        # Check cache
        cached = self.cache.get("find_person", self.last_frame_hash)
        if cached:
            stats.cached_hits += 1
            return cached
        
        print(f"[Vision] 👤 Looking for people...")
        result = self._call_vision_api(VISION_PROMPTS["find_person"], jpeg_bytes)
        
        if result:
            try:
                json_str = result
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0]
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0]
                
                parsed = json.loads(json_str.strip())
                self.cache.set("find_person", self.last_frame_hash, parsed)
                return parsed
            except json.JSONDecodeError:
                return {"raw_response": result, "person_detected": None}
        
        return None
    
    def identify_object(self, target: str) -> Optional[dict]:
        """Look for a specific object in the camera view."""
        jpeg_bytes = self.capture_frame()
        if not jpeg_bytes:
            return None
        
        cache_key = f"identify:{target}"
        cached = self.cache.get(cache_key, self.last_frame_hash)
        if cached:
            stats.cached_hits += 1
            return cached
        
        print(f"[Vision] 🔎 Looking for '{target}'...")
        prompt = VISION_PROMPTS["identify"].format(target=target)
        result = self._call_vision_api(prompt, jpeg_bytes)
        
        if result:
            try:
                json_str = result
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0]
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0]
                
                parsed = json.loads(json_str.strip())
                self.cache.set(cache_key, self.last_frame_hash, parsed)
                return parsed
            except json.JSONDecodeError:
                return {"raw_response": result, "found": None}
        
        return None
    
    def get_exploration_comment(self) -> Optional[str]:
        """Get a curious/playful comment about the scene (for autonomous mode)."""
        jpeg_bytes = self.capture_frame()
        if not jpeg_bytes:
            return None
        
        cached = self.cache.get("explore", self.last_frame_hash)
        if cached:
            stats.cached_hits += 1
            return cached
        
        print(f"[Vision] 🐾 Exploring...")
        result = self._call_vision_api(VISION_PROMPTS["explore"], jpeg_bytes)
        
        if result:
            self.cache.set("explore", self.last_frame_hash, result)
        
        return result
    
    def get_mood(self) -> Optional[str]:
        """Get the mood/atmosphere of the scene."""
        jpeg_bytes = self.capture_frame()
        if not jpeg_bytes:
            return None
        
        cached = self.cache.get("mood", self.last_frame_hash)
        if cached:
            stats.cached_hits += 1
            return cached
        
        print(f"[Vision] 🎭 Analyzing mood...")
        result = self._call_vision_api(VISION_PROMPTS["mood"], jpeg_bytes)
        
        if result:
            self.cache.set("mood", self.last_frame_hash, result)
        
        return result
    
    def custom_query(self, question: str) -> Optional[str]:
        """Ask any custom question about what the camera sees."""
        jpeg_bytes = self.capture_frame()
        if not jpeg_bytes:
            return None
        
        print(f"[Vision] 💬 Custom query: {question[:50]}...")
        return self._call_vision_api(question, jpeg_bytes, max_tokens=500)
    
    def save_snapshot(self, path: Optional[str] = None) -> Optional[str]:
        """Save current frame to file."""
        if self.last_frame is None:
            self.capture_frame()
        
        if self.last_frame is None:
            return None
        
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"/tmp/robot_snapshot_{timestamp}.jpg"
        
        cv2.imwrite(path, self.last_frame)
        print(f"[Vision] 📸 Saved: {path}")
        return path
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "api_calls": stats.api_calls,
            "cached_hits": stats.cached_hits,
            "total_tokens": stats.total_tokens,
            "errors": stats.errors,
            "estimated_cost": f"${stats.estimated_cost:.4f}",
            "cache_entries": len(self.cache.cache)
        }
    
    def release(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("[Vision] 📷 Camera released")


# Singleton instance
_vision: Optional[OpenAIVision] = None


def get_vision() -> OpenAIVision:
    """Get or create the global vision instance."""
    global _vision
    if _vision is None:
        _vision = OpenAIVision()
    return _vision


# ============================================================================
# CLI Testing
# ============================================================================

def main():
    """Test vision capabilities from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI Vision for Robot Pet")
    parser.add_argument("--mode", "-m", default="describe",
                       choices=["describe", "brief", "obstacles", "person", 
                               "explore", "mood", "custom"],
                       help="Query mode")
    parser.add_argument("--target", "-t", help="Object to identify (for identify mode)")
    parser.add_argument("--question", "-q", help="Custom question (for custom mode)")
    parser.add_argument("--snapshot", "-s", action="store_true", help="Save snapshot")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--loop", "-l", type=int, help="Continuous mode (seconds between calls)")
    
    args = parser.parse_args()
    
    vision = get_vision()
    
    if args.stats:
        print("\n📊 Vision Statistics:")
        for k, v in vision.get_stats().items():
            print(f"  {k}: {v}")
        return
    
    def run_query():
        if args.mode == "describe":
            result = vision.describe_scene()
        elif args.mode == "brief":
            result = vision.describe_scene(brief=True)
        elif args.mode == "obstacles":
            result = vision.detect_obstacles()
        elif args.mode == "person":
            result = vision.find_person()
        elif args.mode == "explore":
            result = vision.get_exploration_comment()
        elif args.mode == "mood":
            result = vision.get_mood()
        elif args.mode == "custom" and args.question:
            result = vision.custom_query(args.question)
        elif args.target:
            result = vision.identify_object(args.target)
        else:
            result = vision.describe_scene()
        
        return result
    
    print("\n" + "="*60)
    print("🤖 OpenAI Vision - Robot Pet")
    print("="*60)
    
    if args.loop:
        print(f"Continuous mode: query every {args.loop} seconds")
        print("Press Ctrl+C to stop\n")
        try:
            while True:
                result = run_query()
                print(f"\n📷 Result: {result}")
                print(f"📊 Stats: {vision.get_stats()}")
                if args.snapshot:
                    vision.save_snapshot()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            print("\n\n👋 Stopping...")
    else:
        result = run_query()
        print(f"\n📷 Result:")
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)
        
        if args.snapshot:
            vision.save_snapshot()
        
        print(f"\n📊 Stats: {vision.get_stats()}")
    
    vision.release()


if __name__ == "__main__":
    main()


