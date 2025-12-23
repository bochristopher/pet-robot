#!/usr/bin/env python3
"""
Vision-Only Autonomous Exploration
Uses ONLY OpenAI GPT-4V Vision API for navigation - slow but accurate!

No OpenCV edge detection - just pure AI vision understanding.
Cost: ~$0.60 per 5-minute session (higher but actually works!)

How it works:
1. Capture camera frame
2. Send to GPT-4V with navigation prompt
3. GPT-4V analyzes scene and recommends action
4. Robot executes action
5. Wait, then repeat

Situations handled:
- Clear path → Move forward
- Obstacle ahead → Turn left or right
- Close to wall → Backup then turn
- Cornered → 180° turn
- Person detected → Wave/greet
- Stuck → Try multiple escape strategies
"""

import os
import sys
import time
import random
import cv2
import base64
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motor_interface import get_motors
from elevenlabs_speaker import get_speaker


# =============================================================================
# CONFIGURATION
# =============================================================================

# Movement parameters - CAUTIOUS for small 1ft x 1ft robot
FORWARD_DURATION_CREEP = 0.3  # Tiny creep forward (when obstacle nearby)
FORWARD_DURATION_SHORT = 0.5  # Short forward burst
FORWARD_DURATION_NORMAL = 1.0  # Normal forward movement  
FORWARD_DURATION_LONG = 2.0  # Long forward when VERY clear (wide open space)
TURN_DURATION = 0.8  # Seconds to turn ~45°
TURN_DURATION_90 = 1.2  # Seconds for 90° turn
BACKUP_DURATION = 1.0  # Seconds to back up
ROTATE_180_DURATION = 2.5  # Full 180° turn

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80

# Vision API settings
VISION_API_COST = 0.01  # Cost per Vision API call (~$0.01 for gpt-4o)

# Navigation thresholds
HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.3

# Stuck detection
MAX_CONSECUTIVE_TURNS = 5
MAX_CONSECUTIVE_BACKUPS = 3


# =============================================================================
# NAVIGATION PROMPT - The brain of the robot!
# =============================================================================

NAVIGATION_PROMPT = """You are the navigation AI for a SMALL wheeled robot pet (12 inches tall, 12 inches wide, 12 inches long - about 1 cubic foot).

The camera is mounted at the front of the robot, about 10 inches off the ground.

CRITICAL: This robot is LOW to the ground. Look for obstacles at FLOOR LEVEL:
- Furniture legs (chairs, tables, beds)
- Cables, wires, cords
- Shoes, bags, toys
- Pet bowls, plants
- Walls, doors, cabinets
- Anything below 2 feet tall that could block path

Analyze this camera view and provide a JSON response:

{
    "situation": "wide_open/clear/obstacle_far/obstacle_near/obstacle_close/cornered/person",
    "action": "forward_long/forward/forward_short/creep/turn_left/turn_right/turn_left_90/turn_right_90/backup/rotate_180/stop/greet",
    "confidence": 0.0-1.0,
    "distance_estimate": "far/medium/close/touching",
    "obstacles": ["list obstacles you see"],
    "path_assessment": {
        "left_clear": true/false,
        "center_clear": true/false,
        "right_clear": true/false
    },
    "reason": "what you see and why this action"
}

SITUATIONS (be conservative!):
- "wide_open": Large open space (like middle of room), floor visible 6+ feet ahead, NO obstacles
- "clear": Floor visible 3-6 feet ahead, no immediate obstacles
- "obstacle_far": Something visible 2-3 feet away
- "obstacle_near": Something 1-2 feet away - need to slow down or turn
- "obstacle_close": Something less than 1 foot away - STOP or backup!
- "cornered": Obstacles on multiple sides - need 180° turn
- "person": Human detected

ACTIONS:
- "forward_long": ONLY for wide_open - move 2 seconds (use rarely!)
- "forward": Clear path ahead - move 1 second
- "forward_short": Obstacle_far - move 0.5 seconds cautiously
- "creep": Obstacle_near - inch forward 0.3 seconds, then check again
- "turn_left" / "turn_right": Turn 45° toward clearer side
- "turn_left_90" / "turn_right_90": Turn 90° to avoid obstacle
- "backup": Too close! Back up first
- "rotate_180": Dead end or cornered
- "stop": Something blocking, assess before moving
- "greet": Person detected, be friendly

CRITICAL RULES:
1. DEFAULT TO CAUTION - if obstacle visible, recommend short/creep/turn, NOT forward_long
2. "forward_long" ONLY when you see wide open floor with NOTHING in view
3. If you see ANY furniture, walls, or objects - use forward_short or creep
4. If object fills more than 30% of center frame - recommend turn or backup
5. Look for thin obstacles like chair legs and cables
6. When in doubt, turn to look around rather than move forward

Respond with ONLY the JSON object."""


# =============================================================================
# STATS TRACKING
# =============================================================================

@dataclass
class ExplorationStats:
    """Track exploration session statistics."""
    start_time: float
    movements_forward: int = 0
    movements_creep: int = 0
    turns_left: int = 0
    turns_right: int = 0
    backups: int = 0
    rotations_180: int = 0
    vision_api_calls: int = 0
    speech_chars: int = 0
    greetings: int = 0
    stuck_recoveries: int = 0
    safety_overrides: int = 0

    def __post_init__(self):
        self.start_time = time.time()

    def get_duration(self) -> float:
        return time.time() - self.start_time
    
    @property
    def total_actions(self) -> int:
        return (self.movements_forward + self.movements_creep + self.turns_left + 
                self.turns_right + self.backups + self.rotations_180)

    def get_cost_estimate(self) -> float:
        """Calculate estimated session cost."""
        vision_cost = self.vision_api_calls * VISION_API_COST
        speech_cost = (self.speech_chars / 1000) * 0.03
        return vision_cost + speech_cost


# =============================================================================
# VISION-ONLY EXPLORER
# =============================================================================

class VisionOnlyExplorer:
    """
    Autonomous explorer using ONLY Vision API.
    Slow but reliable - actually understands what it sees!
    """

    def __init__(self):
        print("\n" + "="*60)
        print("🤖 VISION-ONLY EXPLORER")
        print("   Using GPT-4V for intelligent navigation")
        print("="*60)

        self.running = False
        self.exploring = False

        # Initialize modules
        self.motors = get_motors()
        self.speaker = get_speaker()
        self.camera = None
        
        # OpenAI client
        self.client = None
        self._init_openai()
        
        # Tracking
        self.stats = ExplorationStats(start_time=time.time())
        self.consecutive_turns = 0
        self.consecutive_backups = 0
        self.consecutive_forwards = 0  # Track repeated forwards (safety check)
        self.vision_failures = 0  # Track consecutive vision failures
        self.last_action = None
        
        # Voice event flag
        self.voice_describe_requested = False

        print("✅ All modules initialized")
        print("="*60 + "\n")

    def _init_openai(self):
        """Initialize OpenAI client."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[Vision] ❌ OPENAI_API_KEY not set!")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print(f"[Vision] ✅ OpenAI initialized (key: {api_key[:8]}...)")
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

    def _capture_and_encode(self) -> Optional[str]:
        """Capture frame and encode as base64."""
        if not self._init_camera():
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            print("[Vision] ❌ Failed to capture frame")
            return None
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return base64.b64encode(buffer.tobytes()).decode('utf-8')

    def _speak(self, text: str, emotion: str = "curious", blocking: bool = True):
        """Speak with cost tracking."""
        self.stats.speech_chars += len(text)
        self.speaker.speak(text, emotion=emotion, blocking=blocking)

    def _call_vision_api(self, prompt: str, base64_image: str) -> Optional[str]:
        """Call OpenAI Vision API."""
        if not self.client:
            print("[Vision] ❌ OpenAI client not initialized")
            return None
        
        self.stats.vision_api_calls += 1
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # Low detail = faster + cheaper
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Vision] ❌ API error: {e}")
            return None

    def _get_navigation_decision(self) -> Optional[Dict[str, Any]]:
        """
        Ask GPT-4V what to do next.
        
        Returns navigation decision dict or None on failure.
        """
        print(f"[Vision] 🌐 Asking GPT-4V for navigation... (${VISION_API_COST})")
        
        # Capture frame
        base64_image = self._capture_and_encode()
        if not base64_image:
            return None
        
        # Call Vision API
        result = self._call_vision_api(NAVIGATION_PROMPT, base64_image)
        
        if not result:
            return None
        
        # Parse JSON response
        try:
            import json
            # Find JSON in response (sometimes wrapped in markdown)
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = result[start:end]
                decision = json.loads(json_str)
                
                # Print decision
                action = decision.get("action", "unknown")
                confidence = decision.get("confidence", 0.0)
                situation = decision.get("situation", "unknown")
                reason = decision.get("reason", "")[:50]
                
                print(f"[Vision] 📍 Situation: {situation}")
                print(f"[Vision] 🎯 Action: {action} (confidence: {confidence:.2f})")
                print(f"[Vision] 💭 Reason: {reason}...")
                
                return decision
                
        except json.JSONDecodeError as e:
            print(f"[Vision] ⚠️  Failed to parse JSON: {e}")
            print(f"[Vision] Raw response: {result[:100]}...")
        
        return None

    def _execute_action(self, decision: Dict[str, Any]):
        """Execute navigation action based on decision."""
        action = decision.get("action", "stop")
        confidence = decision.get("confidence", 0.0)
        situation = decision.get("situation", "unknown")
        distance = decision.get("distance_estimate", "unknown")
        
        # SAFETY OVERRIDE: If distance is "close" or "touching", don't move forward!
        if distance in ["close", "touching"] and action in ["forward_long", "forward", "forward_short"]:
            print(f"[Vision] 🛡️  SAFETY OVERRIDE: Distance is {distance}, changing forward to turn")
            self.stats.safety_overrides += 1
            action = "turn_right" if decision.get("path_assessment", {}).get("right_clear", True) else "turn_left"
        
        # SAFETY OVERRIDE: If situation is obstacle_close but action is forward, override
        if situation == "obstacle_close" and action in ["forward_long", "forward", "forward_short", "creep"]:
            print(f"[Vision] 🛡️  SAFETY OVERRIDE: Obstacle close, backing up instead")
            self.stats.safety_overrides += 1
            action = "backup"
        
        # Track consecutive actions for stuck detection
        if action in ["turn_left", "turn_right", "turn_left_90", "turn_right_90"]:
            self.consecutive_turns += 1
            self.consecutive_backups = 0
            self.consecutive_forwards = 0
        elif action == "backup":
            self.consecutive_backups += 1
            self.consecutive_turns = 0
            self.consecutive_forwards = 0
        elif action in ["forward_long", "forward", "forward_short", "creep"]:
            self.consecutive_forwards += 1
            self.consecutive_turns = 0
            self.consecutive_backups = 0
        else:
            self.consecutive_turns = 0
            self.consecutive_backups = 0
            self.consecutive_forwards = 0
        
        # SAFETY: If too many consecutive forwards, force a turn to look around
        # This prevents running into things if GPT-4V keeps saying "clear"
        if self.consecutive_forwards >= 4:
            print("[Vision] ⚠️  Too many forwards! Forcing safety turn to look around")
            self.motors.turn_right(TURN_DURATION, blocking=True)
            self.stats.turns_right += 1
            self.consecutive_forwards = 0
            return
        
        # Check if stuck
        if self.consecutive_turns >= MAX_CONSECUTIVE_TURNS:
            print("[Vision] 🔄 Stuck in turning loop! Forcing backup + 180°")
            self._handle_stuck()
            return
        
        if self.consecutive_backups >= MAX_CONSECUTIVE_BACKUPS:
            print("[Vision] 🔄 Stuck backing up! Forcing 180° rotation")
            self.motors.rotate_180(blocking=True)
            self.stats.rotations_180 += 1
            self.consecutive_backups = 0
            return
        
        # Execute based on action
        print(f"[Vision] ⚡ Executing: {action}")
        
        if action == "forward_long":
            # Only for wide open spaces
            self.motors.move_forward(FORWARD_DURATION_LONG, blocking=True)
            self.stats.movements_forward += 1
            
        elif action == "forward":
            self.motors.move_forward(FORWARD_DURATION_NORMAL, blocking=True)
            self.stats.movements_forward += 1
            
        elif action == "forward_short":
            self.motors.move_forward(FORWARD_DURATION_SHORT, blocking=True)
            self.stats.movements_forward += 1
            
        elif action == "creep":
            # Tiny movement when obstacle nearby
            self.motors.move_forward(FORWARD_DURATION_CREEP, blocking=True)
            self.stats.movements_creep += 1
            
        elif action == "turn_left":
            self.motors.turn_left(TURN_DURATION, blocking=True)
            self.stats.turns_left += 1
            
        elif action == "turn_right":
            self.motors.turn_right(TURN_DURATION, blocking=True)
            self.stats.turns_right += 1
            
        elif action == "turn_left_90":
            self.motors.turn_left(TURN_DURATION_90, blocking=True)
            self.stats.turns_left += 1
            
        elif action == "turn_right_90":
            self.motors.turn_right(TURN_DURATION_90, blocking=True)
            self.stats.turns_right += 1
            
        elif action == "backup":
            self.motors.move_backward(BACKUP_DURATION, blocking=True)
            self.stats.backups += 1
            
        elif action == "rotate_180":
            self.motors.rotate_180(blocking=True)
            self.stats.rotations_180 += 1
            
        elif action == "greet":
            self._handle_person()
            
        elif action == "stop":
            self.motors.stop()
            print("[Vision] 🛑 Stopping as recommended")
            time.sleep(1.0)  # Brief pause before trying again
        
        self.last_action = action

    def _handle_stuck(self):
        """Handle being stuck - try escape strategies."""
        self.stats.stuck_recoveries += 1
        print("[Vision] 🆘 Attempting stuck recovery...")
        
        # Backup first
        self.motors.move_backward(BACKUP_DURATION * 1.5, blocking=True)
        time.sleep(0.3)
        
        # 180° turn
        self.motors.rotate_180(blocking=True)
        self.stats.rotations_180 += 1
        
        # Reset counters
        self.consecutive_turns = 0
        self.consecutive_backups = 0

    def _handle_person(self):
        """Handle person detection - be social!"""
        self.stats.greetings += 1
        print("[Vision] 👋 Person detected! Greeting...")
        
        self.motors.stop()
        self._speak("Hello friend!", emotion="excited")
        time.sleep(2.0)

    def set_voice_describe(self):
        """Flag to describe scene on next cycle."""
        self.voice_describe_requested = True

    def _describe_scene(self):
        """Describe current scene to user."""
        print("[Vision] 📢 Describing scene...")
        
        base64_image = self._capture_and_encode()
        if not base64_image:
            self._speak("I can't see right now", emotion="apologetic")
            return
        
        prompt = """Describe what you see in this image in 1-2 short sentences.
Focus on the main objects and their positions. Be concise - this will be spoken aloud."""
        
        result = self._call_vision_api(prompt, base64_image)
        
        if result:
            self._speak(f"I see {result}", emotion="curious")
        else:
            self._speak("I'm having trouble seeing", emotion="apologetic")
        
        self.voice_describe_requested = False

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
            estimated_calls = int(duration / 4)  # ~4 seconds per cycle
            print(f"[Vision] 💰 Estimated cost: ${estimated_calls * VISION_API_COST:.2f}")

        self._speak("Exploring!", emotion="excited")

        try:
            while self.exploring and self.running:
                # Check for voice describe request
                if self.voice_describe_requested:
                    self._describe_scene()
                    continue
                
                # Get navigation decision
                decision = self._get_navigation_decision()
                
                if decision:
                    self.vision_failures = 0  # Reset failure counter
                    confidence = decision.get("confidence", 0.0)
                    
                    if confidence >= LOW_CONFIDENCE:
                        self._execute_action(decision)
                    else:
                        print(f"[Vision] ⚠️  Very low confidence ({confidence:.2f}), turning to look")
                        self.motors.turn_right(TURN_DURATION, blocking=True)
                        self.stats.turns_right += 1
                else:
                    # Vision failed - turn to get new view instead of just stopping
                    self.vision_failures += 1
                    print(f"[Vision] ⚠️  Vision failed ({self.vision_failures}x), turning to look around...")
                    self.motors.turn_right(TURN_DURATION, blocking=True)
                    self.stats.turns_right += 1
                    
                    # If vision keeps failing, stop exploration
                    if self.vision_failures >= 5:
                        print("[Vision] ❌ Too many vision failures, stopping exploration")
                        break
                
                # Brief pause between cycles
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
        self._speak("Done!", emotion="friendly")
        
        # Print stats
        self._print_stats()

    def _print_stats(self):
        """Print session statistics."""
        duration = self.stats.get_duration()
        cost = self.stats.get_cost_estimate()

        print("\n" + "="*60)
        print("📊 VISION-ONLY EXPLORATION SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"🚶 Forward movements: {self.stats.movements_forward}")
        print(f"🐢 Creep movements: {self.stats.movements_creep}")
        print(f"⬅️  Left turns: {self.stats.turns_left}")
        print(f"➡️  Right turns: {self.stats.turns_right}")
        print(f"⬇️  Backups: {self.stats.backups}")
        print(f"🔄 180° rotations: {self.stats.rotations_180}")
        print(f"🛡️  Safety overrides: {self.stats.safety_overrides}")
        print(f"👋 Greetings: {self.stats.greetings}")
        print(f"🆘 Stuck recoveries: {self.stats.stuck_recoveries}")
        print(f"🌐 Vision API calls: {self.stats.vision_api_calls} "
              f"(${self.stats.vision_api_calls * VISION_API_COST:.2f})")
        print(f"💬 Speech chars: {self.stats.speech_chars} "
              f"(${self.stats.speech_chars / 1000 * 0.03:.3f})")
        print(f"\n💰 TOTAL COST: ${cost:.3f}")
        print("="*60 + "\n")

        # Save to log file
        import json
        from datetime import datetime
        log_path = f"/home/bo/robot_pet/logs/vision_explore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create logs directory
        os.makedirs("/home/bo/robot_pet/logs", exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump({
                "duration_seconds": round(duration, 1),
                "total_actions": self.stats.total_actions,
                "forward_movements": self.stats.movements_forward,
                "turns_left": self.stats.turns_left,
                "turns_right": self.stats.turns_right,
                "backups": self.stats.backups,
                "rotations_180": self.stats.rotations_180,
                "greetings": self.stats.greetings,
                "stuck_recoveries": self.stats.stuck_recoveries,
                "vision_api_calls": self.stats.vision_api_calls,
                "speech_chars": self.stats.speech_chars,
                "estimated_cost": f"${cost:.3f}"
            }, f, indent=2)
        print(f"📝 Log saved: {log_path}\n")

    def cleanup(self):
        """Clean up resources."""
        self.motors.stop()
        if self.camera:
            self.camera.release()
        print("[Vision] 🧹 Cleanup complete")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Entry point for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Vision-Only Autonomous Explorer")
    parser.add_argument("--duration", "-d", type=float,
                       help="Exploration duration in seconds")
    parser.add_argument("--test", action="store_true",
                       help="Test single vision call")

    args = parser.parse_args()

    explorer = VisionOnlyExplorer()

    if args.test:
        print("\n🧪 Testing single vision call...\n")
        decision = explorer._get_navigation_decision()
        if decision:
            print(f"\n✅ Decision: {decision}")
        else:
            print("\n❌ Failed to get decision")
        explorer.cleanup()
        return

    try:
        explorer.explore(duration=args.duration)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        explorer.cleanup()


if __name__ == "__main__":
    main()
