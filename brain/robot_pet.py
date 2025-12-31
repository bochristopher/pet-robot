#!/usr/bin/env python3
"""
Robot Pet - Main Integration Loop
Combines voice, vision, speech, movement, and AI personality.

Usage:
    python3 robot_pet.py

Say "hey robot" to wake, then give commands like:
    - "what do you see"
    - "move forward"
    - "turn around"
    - "explore the room"
    - "stop"
"""

import os
import sys
import time
import signal
import threading
from typing import Optional

# Add robot_pet root to path for imports
sys.path.insert(0, '/home/bo/robot_pet')

# Import robot modules using package structure
from voice.whisper_listener import WhisperListener as VoiceListener, get_listener
from voice.elevenlabs_speaker import ElevenLabsSpeaker, get_speaker
from perception.openai_vision import OpenAIVision, get_vision
from actuators.motor_interface import MotorInterface, get_motors
from brain.robot_brain import RobotBrain, get_brain, BrainResponse
from scripts.vision_only_explore import VisionOnlyExplorer


# =============================================================================
# CONFIGURATION
# =============================================================================

# Startup greeting
STARTUP_GREETING = "Ready! Say: hey robot."

# Command patterns for quick actions (bypass AI for speed)
QUICK_COMMANDS = {
    "stop": {"action": "stop"},
    "halt": {"action": "stop"},
    "forward": {"action": "move", "direction": "forward", "duration": 2},
    "go forward": {"action": "move", "direction": "forward", "duration": 2},
    "move forward": {"action": "move", "direction": "forward", "duration": 2},
    "backward": {"action": "move", "direction": "backward", "duration": 2},
    "go back": {"action": "move", "direction": "backward", "duration": 2},
    "move back": {"action": "move", "direction": "backward", "duration": 2},
    "turn left": {"action": "move", "direction": "left", "duration": 1},
    "left": {"action": "move", "direction": "left", "duration": 1},
    "turn right": {"action": "move", "direction": "right", "duration": 1},
    "right": {"action": "move", "direction": "right", "duration": 1},
    "turn around": {"action": "rotate"},
    "rotate": {"action": "rotate"},
    "spin": {"action": "rotate"},
}

# Commands that need vision
VISION_COMMANDS = [
    "what do you see", "what you see", "what do i see", "what can you see",
    "what's around", "look around", "describe", "what's in front",
    "see anything", "what's there", "look", "can you see", "tell me what you see"
]

# Commands that start exploration
EXPLORE_COMMANDS = ["explore", "wander", "look around", "roam"]


class RobotPet:
    """
    Main robot pet controller integrating all modules.
    """
    
    def __init__(self):
        print("\n" + "="*60)
        print("🤖 ROBOT PET - SPARK")
        print("="*60)

        self.running = False
        self.exploring = False
        self.explore_thread: Optional[threading.Thread] = None

        # Initialize modules
        print("\n📦 Loading modules...")

        self.listener = get_listener()
        self.speaker = get_speaker()
        self.vision = get_vision()
        self.motors = get_motors()
        self.brain = get_brain()
        self.explorer: Optional[VisionOnlyExplorer] = None  # Lazy init - Vision API only

        # Set up callbacks
        self.listener.set_wake_callback(self._on_wake)
        self.listener.set_command_callback(self._on_command)

        print("\n✅ All modules loaded!")
        print("="*60)

    def _speak(self, text: str, emotion: str = "friendly", blocking: bool = True):
        """Speak with automatic listener pause/resume to prevent self-hearing."""
        # Pause listener to avoid hearing ourselves
        self.listener.pause()

        # Speak
        self.speaker.speak(text, emotion=emotion, blocking=blocking)

        # Wait a bit for audio to finish if non-blocking
        if not blocking:
            time.sleep(1.0)

        # Resume listener
        time.sleep(0.5)  # Extra buffer
        self.listener.resume()
    
    def _on_wake(self):
        """Called when wake word is detected."""
        # Play a quick acknowledgment sound
        self._speak("Yes?", emotion="curious", blocking=False)
    
    def _on_command(self, command: str):
        """Called when a command is transcribed."""
        self._process_command(command)
    
    def _check_quick_command(self, command: str) -> Optional[dict]:
        """Check if command matches a quick action."""
        cmd_lower = command.lower().strip()
        
        # Direct match
        if cmd_lower in QUICK_COMMANDS:
            return QUICK_COMMANDS[cmd_lower]
        
        # Partial match
        for key, action in QUICK_COMMANDS.items():
            if key in cmd_lower:
                return action
        
        return None
    
    def _needs_vision(self, command: str) -> bool:
        """Check if command requires vision."""
        cmd_lower = command.lower()
        return any(vc in cmd_lower for vc in VISION_COMMANDS)
    
    def _is_explore_command(self, command: str) -> bool:
        """Check if command starts exploration."""
        cmd_lower = command.lower()
        return any(ec in cmd_lower for ec in EXPLORE_COMMANDS)
    
    def _execute_action(self, action: dict):
        """Execute a parsed action."""
        action_type = action.get("type", action.get("action"))
        
        if action_type == "move":
            direction = action.get("direction", "forward")
            duration = action.get("duration", 2)
            
            if direction == "forward":
                self.motors.move_forward(duration, blocking=False)
            elif direction == "backward":
                self.motors.move_backward(duration, blocking=False)
            elif direction == "left":
                self.motors.turn_left(duration, blocking=False)
            elif direction == "right":
                self.motors.turn_right(duration, blocking=False)
                
        elif action_type == "rotate":
            self.motors.rotate_180(blocking=False)
            
        elif action_type == "stop":
            self.motors.stop()
            self._stop_exploration()
            
        elif action_type == "look":
            description = self.vision.describe_scene()
            if description:
                self.brain.set_vision_context(description)
                response = self.brain.think(f"Describe what you see: {description}")
                self.speaker.speak(response.text, emotion=response.emotion)
                
        elif action_type == "explore":
            self._start_exploration()
            
        elif action_type == "follow":
            self._follow_person()
    
    def _process_command(self, command: str):
        """Process a voice command."""
        print(f"\n[Pet] 📝 Processing: '{command}'")

        # Stop exploration if running
        if command.lower() in ["stop", "halt"]:
            self._stop_exploration()
            self.motors.stop()
            self._speak("Stopping!", emotion="calm")
            return

        # Handle vision commands during exploration
        if self.exploring and self._needs_vision(command):
            if self.explorer:
                self.explorer.set_voice_describe()
            return
        
        # Check for quick commands
        quick = self._check_quick_command(command)
        if quick:
            action_type = quick.get("action", quick.get("type"))
            
            # Quick verbal response
            if action_type == "move":
                self._speak("On it!", emotion="playful", blocking=False)
            elif action_type == "rotate":
                self._speak("Wheee!", emotion="excited", blocking=False)
            elif action_type == "stop":
                self._speak("Stopping!", emotion="calm")

            self._execute_action(quick)
            return

        # Check for exploration
        if self._is_explore_command(command):
            # Short response to avoid voice loop
            self._speak("OK!", emotion="excited")
            self._start_exploration()
            return

        # Check for vision commands
        if self._needs_vision(command):
            self._speak("Let me look...", emotion="curious", blocking=True)

            description = self.vision.describe_scene()
            if description:
                self.brain.set_vision_context(description)
                response = self.brain.think(f"I see: {description}. Describe this excitedly.")
                self._speak(response.text, emotion="excited")
            else:
                self._speak("Hmm, I'm having trouble seeing right now.", emotion="apologetic")
            return

        # Use AI brain for complex commands
        response = self.brain.think(command)

        # Speak the response
        self._speak(response.text, emotion=response.emotion, blocking=False)
        
        # Execute any actions
        for action in response.actions:
            self._execute_action(action)
    
    def _start_exploration(self):
        """Start autonomous exploration mode."""
        if self.exploring:
            return

        # Initialize explorer if needed
        if not self.explorer:
            print("[Pet] 🔧 Initializing vision-only explorer (GPT-4V)...")
            self.explorer = VisionOnlyExplorer()

        self.exploring = True
        self.explore_thread = threading.Thread(target=self._explore_loop, daemon=True)
        self.explore_thread.start()
        print("[Pet] 🚀 Exploration started")
    
    def _stop_exploration(self):
        """Stop autonomous exploration."""
        self.exploring = False
        if self.explorer:
            self.explorer.stop()
        print("[Pet] 🛑 Exploration stopped")
    
    def _explore_loop(self):
        """Autonomous exploration loop using VisionOnlyExplorer (GPT-4V)."""
        try:
            # Run autonomous exploration indefinitely
            self.explorer.explore(duration=None)
        except Exception as e:
            print(f"[Pet] ❌ Exploration error: {e}")
            self.exploring = False
    
    def _follow_person(self):
        """Follow a detected person."""
        person = self.vision.find_person()

        if not person or not person.get("person_detected"):
            self._speak("I don't see anyone to follow.", emotion="apologetic")
            return

        position = person.get("positions", ["center"])[0]

        if position == "left":
            self.motors.turn_left(0.3)
        elif position == "right":
            self.motors.turn_right(0.3)

        self.motors.move_forward(1.0)
        self._speak("Following you!", emotion="playful")
    
    def start(self):
        """Start the robot pet."""
        self.running = True

        # Start voice listener first
        self.listener.start()

        # Now speak startup greeting (will pause/resume listener)
        self._speak(STARTUP_GREETING, emotion="friendly")

        print("\n🎤 Listening for 'hey robot'...")
        print("   Say commands like:")
        print("   - 'what do you see'")
        print("   - 'move forward'")
        print("   - 'explore the room'")
        print("   - 'stop'")
        print("\n   Press Ctrl+C to quit\n")
    
    def stop(self):
        """Stop the robot pet."""
        print("\n[Pet] 🛑 Shutting down...")

        self.running = False
        self._stop_exploration()
        self.motors.stop()

        # Say goodbye before stopping listener
        self._speak("Goodbye! See you later!", emotion="friendly")

        self.listener.stop()
        self.vision.release()

        # Cleanup explorer
        if self.explorer:
            self.explorer.cleanup()

        # Print stats
        print("\n📊 Session Statistics:")
        print(f"   Voice: {self.listener.get_stats()}")
        print(f"   Vision: {self.vision.get_stats()}")
        print(f"   Speaker: {self.speaker.get_stats()}")
        print(f"   Motors: {self.motors.get_stats()}")
        print(f"   Brain: {self.brain.get_stats()}")
    
    def run(self):
        """Main run loop."""
        self.start()
        
        try:
            while self.running:
                # Main thread just waits - callbacks handle everything
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robot Pet - Spark")
    parser.add_argument("--test-voice", action="store_true", help="Test voice only")
    parser.add_argument("--test-vision", action="store_true", help="Test vision only")
    parser.add_argument("--test-speak", help="Test speaking a phrase")
    parser.add_argument("--test-move", help="Test movement command")
    
    args = parser.parse_args()
    
    if args.test_voice:
        listener = get_listener()
        listener.start()
        print("Testing voice... say 'hey robot' followed by a command")
        try:
            while True:
                cmd = listener.get_command(timeout=1.0)
                if cmd:
                    print(f"Got: {cmd}")
        except KeyboardInterrupt:
            listener.stop()
        return
    
    if args.test_vision:
        vision = get_vision()
        desc = vision.describe_scene()
        print(f"Vision: {desc}")
        vision.release()
        return
    
    if args.test_speak:
        speaker = get_speaker()
        speaker.speak(args.test_speak, emotion="friendly")
        return
    
    if args.test_move:
        motors = get_motors()
        if args.test_move == "forward":
            motors.move_forward(2)
        elif args.test_move == "backward":
            motors.move_backward(2)
        elif args.test_move == "left":
            motors.turn_left()
        elif args.test_move == "right":
            motors.turn_right()
        motors.stop()
        return
    
    # Normal operation
    pet = RobotPet()
    pet.run()


if __name__ == "__main__":
    main()


