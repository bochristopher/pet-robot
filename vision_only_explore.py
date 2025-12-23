#!/usr/bin/env python3
"""
Vision-Only Autonomous Exploration
Uses ONLY OpenAI GPT-4V Vision API for navigation - slow but accurate!

No OpenCV edge detection - just pure AI vision understanding.
Cost: ~$0.60 per 5-minute session (much higher but actually works!)
"""

import os
import sys
import time
import random
from typing import Optional
from dataclasses import dataclass

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motor_interface import get_motors
from openai_vision import get_vision
from elevenlabs_speaker import get_speaker


# =============================================================================
# CONFIGURATION
# =============================================================================

# Movement parameters
FORWARD_DURATION = 2.0  # Seconds to move forward
TURN_DURATION = 1.0  # Seconds to turn
BACKUP_DURATION = 1.0  # Seconds to back up

# Vision check interval
VISION_CHECK_INTERVAL = 3.0  # Seconds - give Vision API time to respond

# Navigation prompt
NAVIGATION_PROMPT = """You are helping a small wheeled robot navigate a room safely.

Analyze this camera view and respond with ONLY a JSON object:
{
    "path_clear": true/false,
    "action": "forward/left/right/backup/stop",
    "confidence": 0.0-1.0,
    "reason": "brief explanation of what you see and why this action"
}

Rules:
- Look for obstacles at ground level (furniture legs, walls, objects)
- "forward" only if path is completely clear ahead
- "left" or "right" to turn away from obstacles
- "backup" if very close to something
- "stop" if stuck or unsafe
- Be cautious - safety first!"""

# Cost tracking
VISION_API_COST = 0.01  # Cost per Vision API call


@dataclass
class ExplorationStats:
    """Track exploration session statistics."""
    start_time: float
    movements: int = 0
    turns: int = 0
    backups: int = 0
    vision_api_calls: int = 0
    speech_chars: int = 0

    def __post_init__(self):
        self.start_time = time.time()

    def get_duration(self) -> float:
        return time.time() - self.start_time

    def get_cost_estimate(self) -> float:
        """Calculate estimated session cost."""
        vision_cost = self.vision_api_calls * VISION_API_COST
        speech_cost = (self.speech_chars / 1000) * 0.03
        return vision_cost + speech_cost


class VisionOnlyExplorer:
    """
    Autonomous explorer using ONLY Vision API.
    Slow but reliable - actually understands what it sees!
    """

    def __init__(self):
        print("\n" + "="*60)
        print("🤖 VISION-ONLY EXPLORER - Starting...")
        print("="*60)

        self.running = False
        self.exploring = False

        # Initialize modules
        self.motors = get_motors()
        self.vision = get_vision()
        self.speaker = get_speaker()

        # Stats
        self.stats = ExplorationStats(start_time=time.time())

        print("✅ All modules initialized")
        print("="*60 + "\n")

    def _speak(self, text: str, emotion: str = "curious"):
        """Speak with cost tracking."""
        self.stats.speech_chars += len(text)
        self.speaker.speak(text, emotion=emotion, blocking=True)

    def _get_navigation_decision(self) -> Optional[dict]:
        """
        Ask GPT-4V what to do next.

        Returns:
            {
                "path_clear": bool,
                "action": "forward/left/right/backup/stop",
                "confidence": float,
                "reason": str
            }
        """
        self.stats.vision_api_calls += 1
        print(f"[Vision] 🌐 Asking GPT-4V for navigation advice (${VISION_API_COST})...")

        try:
            # Use detect_obstacles method
            result = self.vision.detect_obstacles()

            if result and isinstance(result, dict):
                # Convert to our format
                recommended_action = result.get("recommended_action", "stop")
                confidence = result.get("confidence", 0.5)
                obstacles = result.get("obstacles", [])
                path_clear = result.get("path_clear", False)

                # Build reason string
                if obstacles:
                    reason = f"Obstacles: {', '.join(obstacles[:2])}"
                else:
                    reason = "Path appears clear"

                decision = {
                    "path_clear": path_clear,
                    "action": recommended_action,
                    "confidence": confidence,
                    "reason": reason
                }

                print(f"[Vision] 🤔 Decision: {decision['action']} | "
                      f"Confidence: {decision['confidence']:.2f} | "
                      f"{decision['reason']}")

                return decision

        except Exception as e:
            print(f"[Vision] ❌ Error: {e}")

        return None

    def _execute_action(self, action: str):
        """Execute navigation action."""
        print(f"[Vision] 🎯 Executing: {action}")

        if action == "forward":
            self.stats.movements += 1
            self.motors.move_forward(FORWARD_DURATION, blocking=True)

        elif action == "left":
            self.stats.turns += 1
            self.motors.turn_left(TURN_DURATION, blocking=True)

        elif action == "right":
            self.stats.turns += 1
            self.motors.turn_right(TURN_DURATION, blocking=True)

        elif action == "backup":
            self.stats.backups += 1
            self.motors.move_backward(BACKUP_DURATION, blocking=True)

        elif action == "stop":
            self.motors.stop()
            print("[Vision] 🛑 Stopping as advised by Vision API")

    def explore(self, duration: Optional[float] = None):
        """
        Start vision-only exploration.

        Args:
            duration: Exploration time in seconds (None = indefinite)
        """
        self.exploring = True
        self.running = True

        start_time = time.time()

        print(f"\n[Vision] 🚀 Starting vision-only exploration...")
        if duration:
            print(f"[Vision] ⏱️  Duration: {duration}s")
            print(f"[Vision] 💰 Estimated cost: ${(duration / VISION_CHECK_INTERVAL) * VISION_API_COST:.2f}")

        self._speak("Exploring with AI vision!", emotion="excited")

        try:
            while self.exploring and self.running:
                # Get navigation decision from Vision API
                decision = self._get_navigation_decision()

                if decision:
                    action = decision.get("action", "stop")
                    confidence = decision.get("confidence", 0.0)

                    # Only execute if reasonably confident
                    if confidence > 0.5:
                        self._execute_action(action)

                        # Stop if Vision API says to
                        if action == "stop":
                            print("[Vision] 🛑 Vision API recommends stopping")
                            break
                    else:
                        print(f"[Vision] ⚠️  Low confidence ({confidence:.2f}), turning cautiously")
                        self.motors.turn_right(TURN_DURATION, blocking=True)
                else:
                    # Vision API failed - turn cautiously
                    print("[Vision] ⚠️  Vision API failed, turning cautiously")
                    self.motors.turn_right(TURN_DURATION, blocking=True)

                # Pause before next check
                time.sleep(0.5)

                # Check duration
                if duration and (time.time() - start_time) > duration:
                    print(f"[Vision] ⏱️  Duration reached")
                    break

        except KeyboardInterrupt:
            print("\n[Vision] ⚠️  Interrupted")
        finally:
            self.stop()

    def stop(self):
        """Stop exploration."""
        self.exploring = False
        self.running = False
        self.motors.stop()

        print(f"\n[Vision] 🛑 Exploration stopped")

        self._speak("Done exploring!", emotion="friendly")

        # Print stats
        self._print_stats()

    def _print_stats(self):
        """Print session statistics."""
        duration = self.stats.get_duration()
        cost = self.stats.get_cost_estimate()

        print("\n" + "="*60)
        print("📊 VISION-ONLY EXPLORATION SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {duration:.1f}s")
        print(f"🚶 Forward movements: {self.stats.movements}")
        print(f"🔄 Turns: {self.stats.turns}")
        print(f"⬇️  Backups: {self.stats.backups}")
        print(f"🌐 Vision API calls: {self.stats.vision_api_calls} "
              f"(${self.stats.vision_api_calls * VISION_API_COST:.2f})")
        print(f"💬 Speech chars: {self.stats.speech_chars} "
              f"(${self.stats.speech_chars / 1000 * 0.03:.3f})")
        print(f"\n💰 TOTAL COST: ${cost:.3f}")
        print("="*60 + "\n")

        # Save to log file
        import json
        from datetime import datetime
        log_path = f"/home/bo/robot_pet/vision_explore_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, 'w') as f:
            json.dump({
                "duration_seconds": round(duration, 1),
                "movements": self.stats.movements,
                "turns": self.stats.turns,
                "backups": self.stats.backups,
                "vision_api_calls": self.stats.vision_api_calls,
                "speech_chars": self.stats.speech_chars,
                "estimated_cost": f"${cost:.3f}"
            }, f, indent=2)
        print(f"📝 Log saved: {log_path}\n")

    def cleanup(self):
        """Clean up resources."""
        self.motors.stop()
        print("[Vision] 🧹 Cleanup complete")


def main():
    """Entry point for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Vision-Only Autonomous Explorer")
    parser.add_argument("--duration", "-d", type=float,
                       help="Exploration duration in seconds")

    args = parser.parse_args()

    explorer = VisionOnlyExplorer()

    try:
        explorer.explore(duration=args.duration)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        explorer.cleanup()


if __name__ == "__main__":
    main()
