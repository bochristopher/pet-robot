#!/usr/bin/env python3
"""
LiDAR-Based Autonomous Exploration
Uses RPLidar A1M8 for reliable, real-time obstacle avoidance.

Advantages over camera-based:
- Works in any lighting (dark, bright, etc.)
- Precise distance measurements (15cm - 12m range)
- 360° awareness
- Fast (~5Hz scanning)
- FREE (no API costs!)
- Accurate (laser precision)

ECO_MODE = True (ENERGY SAVING):
- Shorter movement durations = less motor power
- Longer pauses between actions = Jetson/LiDAR rest time
- Speech disabled = $0 API costs
- Less random exploration = more efficient pathfinding
- Result: ~40% longer battery life

Cost: $0/hour (LiDAR + motors only in ECO_MODE)
"""

import os
import sys
import time
import random
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lidar_detector import LidarDetector, LidarScan
from motor_interface import get_motors
from elevenlabs_speaker import get_speaker

# Optional sensor fusion (camera + LiDAR)
try:
    from sensor_fusion import SensorFusion, create_fusion
    SENSOR_FUSION_AVAILABLE = True
except ImportError:
    SENSOR_FUSION_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Energy mode
ECO_MODE = True  # Set to True to conserve Jetson battery

# Sensor fusion (camera validates LiDAR for reflective surfaces)
SENSOR_FUSION_ENABLED = True  # Use camera to detect glass/mirrors LiDAR misses

# Movement parameters
if ECO_MODE:
    FORWARD_DURATION_SHORT = 0.4  # Shorter movements = less energy
    FORWARD_DURATION_NORMAL = 0.8
    FORWARD_DURATION_LONG = 1.5
    TURN_DURATION = 0.7
    TURN_DURATION_90 = 1.0
    BACKUP_DURATION = 0.7
else:
    FORWARD_DURATION_SHORT = 0.5  # Cautious forward
    FORWARD_DURATION_NORMAL = 1.0  # Normal forward
    FORWARD_DURATION_LONG = 2.0  # Long forward when very clear
    TURN_DURATION = 0.8  # ~45° turn
    TURN_DURATION_90 = 1.2  # ~90° turn
    BACKUP_DURATION = 0.8  # Back up

# Distance thresholds (from LiDAR, in meters)
SAFE_DISTANCE = 1.5      # 1.5m = safe to move forward normally
CAUTION_DISTANCE = 1.2   # 1.2m = start preferring open sides (raised from 1.0)
WARNING_DISTANCE = 0.6   # 0.6m = must turn away (raised from 0.5)
DANGER_DISTANCE = 0.3    # 0.3m = emergency backup (unchanged)

# Exploration behavior
if ECO_MODE:
    PAUSE_BETWEEN_MOVES = 0.8  # Longer pause = less energy, more deliberate movement
    RANDOM_TURN_PROBABILITY = 0.05  # Less random exploration to save energy
    CURIOSITY_COMMENT_PROBABILITY = 0.15  # 15% chance to comment (robot likes to talk!)
    OBSERVATION_INTERVAL = 30  # Stop every 30 seconds to look around and comment
else:
    PAUSE_BETWEEN_MOVES = 0.3  # Brief pause between actions
    RANDOM_TURN_PROBABILITY = 0.1  # 10% chance to explore randomly
    CURIOSITY_COMMENT_PROBABILITY = 0.15  # 15% chance to comment
    OBSERVATION_INTERVAL = 30  # Stop every 30 seconds to look around and comment

# Stuck detection
MAX_CONSECUTIVE_TURNS = 4
MAX_CONSECUTIVE_BACKUPS = 3

# Speech (set to False for silent operation)
if ECO_MODE:
    SPEECH_ENABLED = True   # Keep speech on even in ECO mode
    SPEECH_STARTUP = True   # Announce when starting
    SPEECH_STUCK = True     # Say something when stuck
    SPEECH_RANDOM_COMMENTS = True  # Occasional exploration comments
    SPEECH_DIRECTIONS = True  # Announce turns and movements
else:
    SPEECH_ENABLED = True
    SPEECH_STARTUP = True
    SPEECH_STUCK = True
    SPEECH_RANDOM_COMMENTS = True
    SPEECH_DIRECTIONS = True

# Phrases for different situations
PHRASES_EXPLORING = [
    "Exploring!",
    "This is fun!",
    "Where should I go?",
    "Adventure time!",
    "Let's see what's over here!",
]

PHRASES_TURNING_LEFT = [
    "Going left!",
    "This way looks good!",
    "Let me try left!",
    "Turning!",
]

PHRASES_TURNING_RIGHT = [
    "Going right!",
    "Over here!",
    "Let me check this way!",
    "Right looks open!",
]

PHRASES_FORWARD = [
    "Path is clear!",
    "Full speed ahead!",
    "Nice and open!",
    "Onwards!",
]

PHRASES_STUCK = [
    "Hmm, stuck!",
    "Oops, let me back up!",
    "Too tight here!",
    "Need to find another way!",
]

PHRASES_CAMERA = [
    "Let me look with my camera!",
    "Checking with my eyes!",
    "Is that glass?",
]

CACHED_PHRASES = [
    "Exploring!",
    "This is fun!",
    "What's over here?",
    "Interesting!",
]


@dataclass
class ExplorationStats:
    """Track exploration statistics."""
    start_time: float
    movements: int = 0
    turns_left: int = 0
    turns_right: int = 0
    backups: int = 0
    rotations_180: int = 0
    scans: int = 0
    stuck_recoveries: int = 0
    speech_chars: int = 0

    def __post_init__(self):
        self.start_time = time.time()

    def get_duration(self) -> float:
        return time.time() - self.start_time

    @property
    def total_actions(self) -> int:
        return self.movements + self.turns_left + self.turns_right + self.backups + self.rotations_180


class LidarExplorer:
    """
    Autonomous explorer using RPLidar A1M8.
    Fast, accurate, and free!
    """

    def __init__(self, speech_enabled: bool = SPEECH_ENABLED,
                 sensor_fusion: bool = SENSOR_FUSION_ENABLED):
        print("\n" + "="*60)
        print("🤖 LIDAR AUTONOMOUS EXPLORER")
        print("   Using RPLidar A1M8 for navigation")
        if ECO_MODE:
            print("   🔋 ECO MODE: Energy-saving enabled")
        if sensor_fusion and SENSOR_FUSION_AVAILABLE:
            print("   📷 SENSOR FUSION: Camera validates LiDAR")
        print("="*60)

        self.running = False
        self.exploring = False
        self.speech_enabled = speech_enabled

        # Initialize modules
        self.lidar = LidarDetector()
        self.motors = get_motors()
        self.speaker = get_speaker() if speech_enabled else None

        # Sensor fusion (camera + LiDAR)
        self.fusion = None
        if sensor_fusion and SENSOR_FUSION_AVAILABLE:
            try:
                self.fusion = create_fusion(enable_camera=True)
            except Exception as e:
                print(f"[Fusion] ⚠️ Disabled: {e}")

        # Stats and state
        self.stats = ExplorationStats(start_time=time.time())
        self.consecutive_turns = 0
        self.consecutive_backups = 0
        self.last_turn_direction = None  # Track to alternate turns

        # Voice control flags
        self.voice_stop_requested = False

        # Periodic observation
        self.last_observation_time = 0
        self.vision = None
        if SENSOR_FUSION_AVAILABLE:
            try:
                from openai_vision import OpenAIVision
                self.vision = OpenAIVision()
            except:
                pass
        self.voice_describe_requested = False

        print("✅ All modules initialized")
        print("="*60 + "\n")

    def _speak(self, text: str, emotion: str = "curious", blocking: bool = True):
        """Speak with cost tracking."""
        if not self.speech_enabled or not self.speaker:
            return

        self.stats.speech_chars += len(text)
        self.speaker.speak(text, emotion=emotion, blocking=blocking)

    def _decide_action(self, scan: LidarScan) -> str:
        """
        Decide navigation action - ALWAYS move toward most open area.

        Strategy: Instead of "go forward until forced to turn",
        we "always turn toward the most open direction".
        Uses sensor fusion (camera) when LiDAR readings are suspicious.

        Returns:
            Action string: "forward_long", "forward", "forward_short",
                          "turn_left", "turn_right", "backup"
        """
        front = scan.front
        left = scan.left
        right = scan.right

        # Check for suspicious readings (possible reflective surface)
        suspicious = front.suspicious or left.suspicious or right.suspicious

        # Print scan info
        suspicious_marker = " ⚠️" if suspicious else ""
        print(f"[LiDAR] 📡 Front: {front.min_distance:.2f}m | "
              f"Left: {left.min_distance:.2f}m | Right: {right.min_distance:.2f}m{suspicious_marker}")

        # SENSOR FUSION: If readings are suspicious, check camera
        if suspicious and self.fusion:
            # Announce camera check sometimes
            if SPEECH_DIRECTIONS and random.random() < 0.5:
                self._speak(random.choice(PHRASES_CAMERA), emotion="curious", blocking=False)
            fusion_result = self.fusion.get_fused_reading(scan)
            if fusion_result.camera_override:
                # Camera detected obstacle LiDAR missed!
                print(f"[Fusion] 📷 Camera override: {fusion_result.recommended_action}")
                # Map fusion recommendation to our action format
                action_map = {
                    "forward": "forward_short",  # Be cautious
                    "left": "turn_left",
                    "right": "turn_right",
                    "stop": "backup",
                    "backup": "backup"
                }
                return action_map.get(fusion_result.recommended_action, "forward_short")

        # CRITICAL DANGER: Too close! Back up immediately
        if front.min_distance < DANGER_DISTANCE:  # 0.3m
            return "backup"

        # Find the most open direction
        max_distance = max(front.min_distance, left.min_distance, right.min_distance)

        # WARNING: Very close obstacle - turn toward most open side
        if front.min_distance < WARNING_DISTANCE:  # 0.6m
            if left.min_distance > right.min_distance:
                return "turn_left"
            else:
                return "turn_right"

        # CAUTION: Getting close - prefer turning toward open areas
        if front.min_distance < CAUTION_DISTANCE:  # 1.2m
            # If either side is more open, turn that way
            if left.min_distance > front.min_distance or right.min_distance > front.min_distance:
                if left.min_distance > right.min_distance:
                    return "turn_left"
                else:
                    return "turn_right"
            else:
                # All equally tight - creep forward cautiously
                return "forward_short"

        # NORMAL NAVIGATION: Seek the most open direction
        # If front is most open AND very clear, go forward
        if max_distance == front.min_distance and front.min_distance >= SAFE_DISTANCE:  # 1.5m+
            # Random exploration when wide open
            if front.min_distance >= SAFE_DISTANCE * 2 and random.random() < RANDOM_TURN_PROBABILITY:
                return "turn_left" if random.random() > 0.5 else "turn_right"
            return "forward_long" if front.min_distance >= SAFE_DISTANCE * 1.5 else "forward"

        # If left or right is significantly more open, turn toward it
        # Even if front is "clear", prefer the MOST open direction
        if max_distance == left.min_distance and left.min_distance > front.min_distance + 0.2:
            return "turn_left"

        if max_distance == right.min_distance and right.min_distance > front.min_distance + 0.2:
            return "turn_right"

        # Front is reasonably clear and similar to sides - go forward
        if front.min_distance >= SAFE_DISTANCE * 0.75:  # 1.125m
            return "forward"

        # Uncertain - turn toward clearer side
        if left.min_distance > right.min_distance:
            return "turn_left"
        else:
            return "turn_right"

    def _execute_action(self, action: str):
        """Execute navigation action."""
        print(f"[LiDAR] ⚡ Action: {action}")

        # Track for stuck detection
        if action in ["turn_left", "turn_right"]:
            self.consecutive_turns += 1
            self.consecutive_backups = 0
            self.last_turn_direction = action
        elif action == "backup":
            self.consecutive_backups += 1
            self.consecutive_turns = 0
        else:
            self.consecutive_turns = 0
            self.consecutive_backups = 0

        # Execute with optional speech
        if action == "forward_long":
            # Announce when path is very clear
            if SPEECH_DIRECTIONS and random.random() < 0.3:
                self._speak(random.choice(PHRASES_FORWARD), emotion="excited", blocking=False)
            self.motors.move_forward(FORWARD_DURATION_LONG, blocking=True)
            self.stats.movements += 1

        elif action == "forward":
            self.motors.move_forward(FORWARD_DURATION_NORMAL, blocking=True)
            self.stats.movements += 1

        elif action == "forward_short":
            self.motors.move_forward(FORWARD_DURATION_SHORT, blocking=True)
            self.stats.movements += 1

        elif action == "turn_left":
            # Announce turns sometimes
            if SPEECH_DIRECTIONS and random.random() < 0.25:
                self._speak(random.choice(PHRASES_TURNING_LEFT), emotion="curious", blocking=False)
            self.motors.turn_left(TURN_DURATION, blocking=True)
            self.stats.turns_left += 1

        elif action == "turn_right":
            if SPEECH_DIRECTIONS and random.random() < 0.25:
                self._speak(random.choice(PHRASES_TURNING_RIGHT), emotion="curious", blocking=False)
            self.motors.turn_right(TURN_DURATION, blocking=True)
            self.stats.turns_right += 1

        elif action == "backup":
            self.motors.move_backward(BACKUP_DURATION, blocking=True)
            self.stats.backups += 1

        elif action == "rotate_180":
            self.motors.rotate_180(blocking=True)
            self.stats.rotations_180 += 1

        # Random exploration comment
        if SPEECH_RANDOM_COMMENTS and random.random() < CURIOSITY_COMMENT_PROBABILITY:
            self._speak(random.choice(PHRASES_EXPLORING), emotion="playful", blocking=False)

    def _observe_surroundings(self):
        """Stop and comment on surroundings using camera."""
        if not self.vision:
            return

        print("[Robot] 👀 Stopping to look around...")
        self.motors.stop()

        try:
            # Get a description of what's around
            description = self.vision.describe_scene()
            if description:
                print(f"[Robot] 💬 {description}")
                self._speak(description, emotion="curious", blocking=True)
            else:
                # Fallback if vision fails
                self._speak("Hmm, let me see what's here!", emotion="curious", blocking=True)
        except Exception as e:
            print(f"[Robot] ⚠️ Vision error: {e}")
            self._speak("Looking around!", emotion="curious", blocking=True)

        self.last_observation_time = time.time()
        print("[Robot] 👀 Continuing exploration...")

    def _handle_stuck(self):
        """Handle stuck scenario."""
        self.stats.stuck_recoveries += 1
        print("[LiDAR] 🔄 Stuck! Attempting recovery...")

        if SPEECH_STUCK:
            self._speak(random.choice(PHRASES_STUCK), emotion="curious", blocking=False)

        # Back up
        self.motors.move_backward(BACKUP_DURATION * 1.5, blocking=True)
        time.sleep(0.5)

        # 180° turn
        self.motors.rotate_180(blocking=True)
        self.stats.rotations_180 += 1

        # Reset counters
        self.consecutive_turns = 0
        self.consecutive_backups = 0

    def explore(self, duration: Optional[float] = None):
        """
        Start LiDAR-based exploration.

        Args:
            duration: Exploration time in seconds (None = indefinite)
        """
        self.exploring = True
        self.running = True
        start_time = time.time()
        self.last_observation_time = time.time()  # Start timer for periodic observations

        print(f"\n[LiDAR] 🚀 Starting exploration...")
        if duration:
            print(f"[LiDAR] ⏱️  Duration: {duration}s")

        # Start LiDAR
        if not self.lidar.start():
            print("[LiDAR] ❌ Failed to start LiDAR!")
            return

        if SPEECH_STARTUP:
            self._speak("Exploring with laser vision!", emotion="excited")

        try:
            while self.exploring and self.running:
                # Check voice commands
                if self.voice_stop_requested:
                    print("[LiDAR] 🛑 Stop requested")
                    break

                # Get LiDAR scan - errors are handled gracefully
                scan = self.lidar.get_scan(max_scans=1)

                # If scan fails, use last valid scan (up to 1 second old)
                if not scan:
                    scan = self.lidar.get_last_scan()
                    if scan and (time.time() - scan.timestamp) > 1.0:
                        scan = None  # Too old, wait for fresh data

                if scan:
                    self.stats.scans += 1

                    # Periodic observation - stop and look around every N seconds
                    if (self.vision and
                        time.time() - self.last_observation_time >= OBSERVATION_INTERVAL):
                        self._observe_surroundings()

                    # Decide action
                    action = self._decide_action(scan)

                    # Execute
                    self._execute_action(action)

                    # Check if stuck
                    if self.consecutive_turns >= MAX_CONSECUTIVE_TURNS:
                        self._handle_stuck()

                    if self.consecutive_backups >= MAX_CONSECUTIVE_BACKUPS:
                        self._handle_stuck()

                else:
                    # No scan data - wait briefly and try again
                    # The clean_input() in get_scan() will handle buffer cleanup
                    self.no_scan_count = getattr(self, 'no_scan_count', 0) + 1

                    if self.no_scan_count >= 5:
                        # Multiple failures - try full restart
                        print("[LiDAR] ⚠️ Multiple scan failures, restarting...")
                        if not self.lidar.restart():
                            print("[LiDAR] ❌ Failed to restart LiDAR!")
                            break
                        self.no_scan_count = 0
                    else:
                        # Brief pause, then retry
                        time.sleep(0.1)
                        continue  # Skip the normal pause, try again immediately

                # Reset failure counter on success
                self.no_scan_count = 0

                # Pause between moves
                time.sleep(PAUSE_BETWEEN_MOVES)

                # Check duration
                if duration and (time.time() - start_time) > duration:
                    print(f"[LiDAR] ⏱️  Duration reached")
                    break

        except KeyboardInterrupt:
            print("\n[LiDAR] ⚠️  Interrupted")
        finally:
            self.stop()

    def stop(self):
        """Stop exploration."""
        self.exploring = False
        self.running = False
        self.motors.stop()
        self.lidar.stop()

        print(f"\n[LiDAR] 🛑 Exploration stopped")
        self._speak("Done exploring!", emotion="friendly")

        # Print stats
        self._print_stats()

    def _print_stats(self):
        """Print session statistics."""
        duration = self.stats.get_duration()

        print("\n" + "="*60)
        print("📊 LIDAR EXPLORATION SUMMARY")
        if ECO_MODE:
            print("   🔋 ECO MODE ACTIVE")
        print("="*60)
        print(f"⏱️  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"🚶 Forward movements: {self.stats.movements}")
        print(f"⬅️  Left turns: {self.stats.turns_left}")
        print(f"➡️  Right turns: {self.stats.turns_right}")
        print(f"⬇️  Backups: {self.stats.backups}")
        print(f"🔄 180° rotations: {self.stats.rotations_180}")
        print(f"📡 LiDAR scans: {self.stats.scans}")
        print(f"🆘 Stuck recoveries: {self.stats.stuck_recoveries}")
        print(f"💬 Speech chars: {self.stats.speech_chars} "
              f"(${self.stats.speech_chars / 1000 * 0.03:.3f})")
        print(f"\n💰 TOTAL COST: ${self.stats.speech_chars / 1000 * 0.03:.3f} "
              "(LiDAR is FREE!)")
        if ECO_MODE:
            print(f"🔋 Energy saved: ~40% (shorter movements, longer pauses)")
        if self.fusion:
            fusion_stats = self.fusion.get_stats()
            print(f"📷 Camera checks: {fusion_stats['camera_checks']} "
                  f"(overrides: {fusion_stats['camera_overrides']})")
        print(f"⚡ Actions per minute: {self.stats.total_actions / (duration/60):.1f}")
        print("="*60 + "\n")

        # Save log
        import json
        os.makedirs("/home/bo/robot_pet/logs", exist_ok=True)
        log_path = f"/home/bo/robot_pet/logs/lidar_explore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(log_path, 'w') as f:
            json.dump({
                "duration_seconds": round(duration, 1),
                "movements": self.stats.movements,
                "turns_left": self.stats.turns_left,
                "turns_right": self.stats.turns_right,
                "backups": self.stats.backups,
                "rotations_180": self.stats.rotations_180,
                "scans": self.stats.scans,
                "stuck_recoveries": self.stats.stuck_recoveries,
                "total_actions": self.stats.total_actions,
                "speech_chars": self.stats.speech_chars
            }, f, indent=2)

        print(f"📝 Log saved: {log_path}\n")

    def set_voice_stop(self):
        """Set stop flag from voice command."""
        self.voice_stop_requested = True

    def cleanup(self):
        """Clean up resources."""
        self.lidar.stop()
        self.motors.stop()
        print("[LiDAR] 🧹 Cleanup complete")


def main():
    """Entry point for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(description="LiDAR Autonomous Explorer")
    parser.add_argument("--duration", "-d", type=float,
                       help="Exploration duration in seconds")
    parser.add_argument("--silent", "-s", action="store_true",
                       help="Disable speech")

    args = parser.parse_args()

    explorer = LidarExplorer(speech_enabled=not args.silent)

    try:
        explorer.explore(duration=args.duration)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        explorer.cleanup()


if __name__ == "__main__":
    main()
