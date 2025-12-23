#!/usr/bin/env python3
"""
Autonomous Exploration Mode - Cost-Optimized for Jetson Orin Nano

Strategy: OpenCV Primary with Vision API Fallback
- OpenCV edge detection for continuous obstacle avoidance (free, 15ms)
- GPT-4V Vision API for stuck scenarios and periodic curiosity (sparse)
- Voice control integration (stop, what do you see)
- Cost tracking and phrase caching

Estimated cost: ~$0.11 per 5-minute session
"""

import os
import sys
import cv2
import time
import random
import json
import threading
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motor_interface import get_motors
from openai_vision import get_vision
from elevenlabs_speaker import get_speaker

# =============================================================================
# CONFIGURATION
# =============================================================================

# Movement parameters
FORWARD_DURATION_MIN = 1.5  # Min seconds to move forward
FORWARD_DURATION_MAX = 3.0  # Max seconds to move forward
TURN_DURATION = 1.0  # Seconds to turn when obstacle detected
PAUSE_BETWEEN_MOVES = 0.5  # Brief pause between movements

# Vision parameters
OBSTACLE_EDGE_THRESHOLD = 25000  # Edge pixels to consider obstacle (raised from 8000)
CENTER_WEIGHT = 2.0  # Weight for center region obstacles
VISION_CHECK_INTERVAL = 2.0  # Seconds between obstacle checks

# Curiosity parameters
CURIOSITY_INTERVAL = 300.0  # 5 minutes between curiosity checks
COMMENT_PROBABILITY = 0.15  # 15% chance to comment after movement

# Stuck detection
STUCK_TURNS_THRESHOLD = 3  # Turns before considering "stuck"
STUCK_VISION_DELAY = 5.0  # Seconds to wait before using Vision API

# Cost tracking
VISION_API_COST = 0.01  # Cost per Vision API call
SPEECH_COST_PER_1000 = 0.03  # ElevenLabs cost per 1000 chars

# Common phrases for caching
CACHED_PHRASES = [
    "Oops, something's there!",
    "Let me turn around",
    "Hmm, what's over here?",
    "Oh, interesting!",
    "I wonder what that is",
    "This is fun!",
    "Exploring is exciting!",
]


@dataclass
class ExplorationStats:
    """Track exploration session statistics."""
    start_time: float = field(default_factory=time.time)
    movements: int = 0
    turns: int = 0
    opencv_checks: int = 0
    vision_api_calls: int = 0
    curiosity_checks: int = 0
    stuck_recoveries: int = 0
    speech_chars: int = 0

    def get_duration(self) -> float:
        return time.time() - self.start_time

    def get_cost_estimate(self) -> float:
        """Calculate estimated session cost."""
        vision_cost = self.vision_api_calls * VISION_API_COST
        speech_cost = (self.speech_chars / 1000) * SPEECH_COST_PER_1000
        return vision_cost + speech_cost

    def to_dict(self) -> dict:
        return {
            "duration_seconds": round(self.get_duration(), 1),
            "movements": self.movements,
            "turns": self.turns,
            "opencv_checks": self.opencv_checks,
            "vision_api_calls": self.vision_api_calls,
            "curiosity_checks": self.curiosity_checks,
            "stuck_recoveries": self.stuck_recoveries,
            "speech_chars": self.speech_chars,
            "estimated_cost": f"${self.get_cost_estimate():.3f}"
        }


class ObstacleDetector:
    """OpenCV-based obstacle detection."""

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("[Obstacle] ✅ Camera initialized (640x480)")

    def detect(self) -> Dict:
        """
        Detect obstacles using edge detection.

        Returns:
            {
                "path_clear": bool,
                "edge_count": int,
                "center_edges": int,
                "processing_time_ms": float
            }
        """
        start_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            return {"path_clear": True, "error": "Failed to read frame"}

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Count edges in center region (where robot will move)
        height, width = edges.shape
        center_x = width // 2
        center_width = width // 3
        center_region = edges[:, center_x - center_width//2 : center_x + center_width//2]

        total_edges = cv2.countNonZero(edges)
        center_edges = cv2.countNonZero(center_region)

        # Weight center more heavily
        weighted_edges = total_edges + (center_edges * CENTER_WEIGHT)

        path_clear = weighted_edges < OBSTACLE_EDGE_THRESHOLD

        processing_time = (time.time() - start_time) * 1000

        return {
            "path_clear": path_clear,
            "edge_count": total_edges,
            "center_edges": center_edges,
            "weighted_edges": int(weighted_edges),
            "processing_time_ms": round(processing_time, 1)
        }

    def release(self):
        """Release camera."""
        if self.cap:
            self.cap.release()
        print("[Obstacle] 🛑 Camera released")


class AutonomousExplorer:
    """
    Autonomous exploration controller.

    Uses OpenCV for obstacle detection, GPT-4V for stuck scenarios
    and periodic curiosity checks.
    """

    def __init__(self):
        print("\n" + "="*60)
        print("🤖 AUTONOMOUS EXPLORER - Starting...")
        print("="*60)

        self.running = False
        self.exploring = False

        # Initialize modules
        self.detector = ObstacleDetector()
        self.motors = get_motors()
        self.vision = get_vision()
        self.speaker = get_speaker()

        # Stats and state
        self.stats = ExplorationStats()
        self.consecutive_turns = 0
        self.last_curiosity_time = time.time()
        self.last_comment_time = time.time()

        # Voice control (set externally)
        self.voice_stop_requested = False
        self.voice_describe_requested = False

        print("✅ All modules initialized")
        print("="*60 + "\n")

    def _speak(self, text: str, emotion: str = "curious", blocking: bool = False):
        """Speak with cost tracking."""
        self.stats.speech_chars += len(text)
        self.speaker.speak(text, emotion=emotion, blocking=blocking)

    def _check_obstacle(self) -> bool:
        """
        Check for obstacles using OpenCV.

        Returns:
            True if path is clear, False if obstacle detected
        """
        self.stats.opencv_checks += 1
        result = self.detector.detect()

        if result.get("error"):
            print(f"[Explorer] ⚠️  Detection error: {result['error']}")
            return True  # Assume clear on error

        path_clear = result["path_clear"]

        # Always show what we're seeing
        status = "✅ CLEAR" if path_clear else "🚧 BLOCKED"
        print(f"[Explorer] 👁️  {status} | "
              f"Total edges: {result['edge_count']:>5} | "
              f"Center edges: {result['center_edges']:>5} | "
              f"Weighted: {result['weighted_edges']:>5} | "
              f"Threshold: {OBSTACLE_EDGE_THRESHOLD} | "
              f"{result['processing_time_ms']:.1f}ms")

        return path_clear

    def _get_vision_description(self) -> Optional[str]:
        """
        Get GPT-4V description of current view.

        Returns:
            Description string or None on error
        """
        self.stats.vision_api_calls += 1
        print(f"[Explorer] 🌐 Calling Vision API (${VISION_API_COST})...")

        # Capture frame from our camera (already open)
        ret, frame = self.detector.cap.read()
        if not ret:
            print("[Explorer] ❌ Failed to capture frame")
            return None

        # Save temp image for Vision API
        import tempfile
        temp_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(temp_path, frame)

        # Get description using our captured frame
        description = self.vision.describe_image(temp_path)

        # Cleanup
        import os
        try:
            os.remove(temp_path)
        except:
            pass

        if description:
            print(f"[Explorer] 📝 Vision: {description}")
        else:
            print("[Explorer] ❌ Vision API failed")

        return description

    def _handle_stuck(self):
        """Handle stuck scenario using Vision API."""
        self.stats.stuck_recoveries += 1
        print(f"[Explorer] 🔄 Stuck detected! Backing up to get better view...")

        # Brief speech to avoid voice loop
        self._speak("Hmm, stuck.", emotion="curious", blocking=True)

        # IMPORTANT: Back up first so camera can see surroundings
        # If we're against a wall, camera just sees blank surface - useless!
        self.motors.move_backward(1.5, blocking=True)

        # Wait for movement to complete and camera to stabilize
        time.sleep(0.5)

        # Now get detailed view from a distance
        print(f"[Explorer] 📸 Taking picture from distance...")
        description = self._get_vision_description()

        if description:
            # Use vision to decide action silently to avoid voice loop
            print(f"[Explorer] 🤔 Vision says: {description}")

            if "wall" in description.lower() or "blocked" in description.lower() or "obstacle" in description.lower():
                # Major obstacle - turn around completely
                self.motors.rotate_180(blocking=True)
                self.consecutive_turns = 0
            elif "open" in description.lower() or "clear" in description.lower() or "space" in description.lower():
                # Seems open - try turning to find path
                if random.random() > 0.5:
                    self.motors.turn_left(TURN_DURATION * 1.5, blocking=True)
                else:
                    self.motors.turn_right(TURN_DURATION * 1.5, blocking=True)
                self.consecutive_turns = 0
            else:
                # Unclear - rotate 180 to try completely different direction
                self.motors.rotate_180(blocking=True)
                self.consecutive_turns = 0
        else:
            # Fallback if Vision API fails - just rotate and try again
            print("[Explorer] ⚠️  Vision failed, rotating to new direction")
            self.motors.rotate_180(blocking=True)
            self.consecutive_turns = 0

    def _curiosity_check(self):
        """Periodic curiosity check using Vision API."""
        self.stats.curiosity_checks += 1
        print(f"[Explorer] 🤔 Curiosity check...")

        # Brief speech
        self._speak("What's here?", emotion="curious", blocking=True)

        description = self._get_vision_description()

        if description:
            # Keep it short
            self._speak(f"I see {description[:50]}", emotion="excited", blocking=True)

        self.last_curiosity_time = time.time()

    def _random_comment(self):
        """Make a random exploratory comment (cached phrases)."""
        if random.random() < COMMENT_PROBABILITY:
            comment = random.choice(CACHED_PHRASES)
            self._speak(comment, emotion="playful", blocking=False)
            self.last_comment_time = time.time()

    def _exploration_step(self):
        """Perform one exploration step."""

        # Check for obstacles
        if not self._check_obstacle():
            # Obstacle detected - turn
            self.stats.turns += 1
            self.consecutive_turns += 1

            # Silent turn - no speech to avoid voice loop
            # Random turn direction
            if random.random() > 0.5:
                self.motors.turn_left(TURN_DURATION, blocking=True)
            else:
                self.motors.turn_right(TURN_DURATION, blocking=True)

            # Check if stuck
            if self.consecutive_turns >= STUCK_TURNS_THRESHOLD:
                time.sleep(STUCK_VISION_DELAY)
                self._handle_stuck()

        else:
            # Path clear - move forward
            self.stats.movements += 1
            self.consecutive_turns = 0  # Reset stuck counter

            duration = random.uniform(FORWARD_DURATION_MIN, FORWARD_DURATION_MAX)
            self.motors.move_forward(duration, blocking=True)

            # No random comments to avoid voice loop

        # Pause between movements
        time.sleep(PAUSE_BETWEEN_MOVES)

        # Periodic curiosity check
        if (time.time() - self.last_curiosity_time) > CURIOSITY_INTERVAL:
            self._curiosity_check()

    def explore(self, duration: Optional[float] = None):
        """
        Start autonomous exploration.

        Args:
            duration: Exploration time in seconds (None = indefinite)
        """
        self.exploring = True
        self.running = True

        start_time = time.time()

        print(f"\n[Explorer] 🚀 Starting exploration...")
        if duration:
            print(f"[Explorer] ⏱️  Duration: {duration}s")

        # Short startup speech to avoid triggering voice loop
        self._speak("Exploring!", emotion="excited", blocking=True)

        try:
            while self.exploring and self.running:
                # Check voice commands
                if self.voice_stop_requested:
                    print("[Explorer] 🛑 Stop requested via voice")
                    break

                if self.voice_describe_requested:
                    print("[Explorer] 👁️  Describe requested via voice")
                    description = self._get_vision_description()
                    if description:
                        self._speak(f"I see {description}", emotion="excited", blocking=True)
                    self.voice_describe_requested = False

                # Exploration step
                self._exploration_step()

                # Check duration
                if duration and (time.time() - start_time) > duration:
                    print(f"[Explorer] ⏱️  Duration reached")
                    break

        except KeyboardInterrupt:
            print("\n[Explorer] ⚠️  Interrupted")
        finally:
            self.stop()

    def stop(self):
        """Stop exploration."""
        self.exploring = False
        self.running = False
        self.motors.stop()

        print(f"\n[Explorer] 🛑 Exploration stopped")

        # Brief end speech
        self._speak("Done!", emotion="friendly", blocking=True)

        # Print stats
        self._print_stats()

    def _print_stats(self):
        """Print session statistics."""
        stats_dict = self.stats.to_dict()

        print("\n" + "="*60)
        print("📊 EXPLORATION SESSION SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {stats_dict['duration_seconds']}s")
        print(f"🚶 Movements: {stats_dict['movements']}")
        print(f"🔄 Turns: {stats_dict['turns']}")
        print(f"👁️  OpenCV checks: {stats_dict['opencv_checks']} (free)")
        print(f"🌐 Vision API calls: {stats_dict['vision_api_calls']} "
              f"(${stats_dict['vision_api_calls'] * VISION_API_COST:.2f})")
        print(f"🤔 Curiosity checks: {stats_dict['curiosity_checks']}")
        print(f"🔄 Stuck recoveries: {stats_dict['stuck_recoveries']}")
        print(f"💬 Speech chars: {stats_dict['speech_chars']} "
              f"(${stats_dict['speech_chars'] / 1000 * SPEECH_COST_PER_1000:.3f})")
        print(f"\n💰 TOTAL COST: {stats_dict['estimated_cost']}")
        print("="*60 + "\n")

        # Save to log file
        log_path = f"/home/bo/robot_pet/exploration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"📝 Log saved: {log_path}\n")

    def set_voice_stop(self):
        """Set stop flag (called from voice command)."""
        self.voice_stop_requested = True

    def set_voice_describe(self):
        """Request description (called from voice command)."""
        self.voice_describe_requested = True

    def cleanup(self):
        """Clean up resources."""
        self.detector.release()
        self.motors.stop()
        print("[Explorer] 🧹 Cleanup complete")


def main():
    """Entry point for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Explorer")
    parser.add_argument("--duration", "-d", type=float,
                       help="Exploration duration in seconds")
    parser.add_argument("--test-obstacle", action="store_true",
                       help="Test obstacle detection only")

    args = parser.parse_args()

    if args.test_obstacle:
        print("\n🧪 Testing obstacle detection...\n")
        detector = ObstacleDetector()

        try:
            for i in range(10):
                result = detector.detect()
                status = "✅ CLEAR" if result["path_clear"] else "🚧 BLOCKED"
                print(f"{i+1}. {status} | Edges: {result['edge_count']} "
                      f"(center: {result['center_edges']}) | "
                      f"{result['processing_time_ms']}ms")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            detector.release()
        return

    # Normal exploration
    explorer = AutonomousExplorer()

    try:
        explorer.explore(duration=args.duration)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        explorer.cleanup()


if __name__ == "__main__":
    main()
