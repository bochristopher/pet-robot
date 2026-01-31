#!/usr/bin/env python3
"""
Motor Interface Module
Wrapper for controlling robot motors via the WebSocket server.

Features:
- WebSocket connection to server.py (port 8765)
- Authentication with token
- High-level movement functions
- Timed movements with automatic stop
- Connection error handling
"""

import os
import sys
import time
import json
import asyncio
import threading
from typing import Optional
from dataclasses import dataclass

# Try to import required libraries
try:
    import websockets
    from websockets.sync.client import connect as ws_connect
except ImportError:
    print("❌ websockets not installed: pip install websockets")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Server settings (WebSocket server on Jetson)
SERVER_HOST = os.environ.get("ROBOT_SERVER_HOST", "localhost")
SERVER_PORT = int(os.environ.get("ROBOT_SERVER_PORT", "8765"))
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}"

# Authentication token
AUTH_TOKEN = os.environ.get("ROBOT_AUTH_TOKEN", "robot_secret_2024")

# Default movement parameters
DEFAULT_SPEED = 0.5  # Not used by current server but kept for future
DEFAULT_DURATION = 2.0  # seconds
TURN_DURATION = 0.8  # seconds for 90° turn
ROTATE_180_DURATION = 1.6  # seconds for 180° turn

# Valid directions
VALID_DIRECTIONS = ["forward", "backward", "left", "right", "stop"]


@dataclass
class MotorStats:
    """Track motor control statistics."""
    commands_sent: int = 0
    successful: int = 0
    failed: int = 0
    total_move_time: float = 0


stats = MotorStats()


class MotorInterface:
    """
    Motor control interface for robot pet using WebSocket.
    
    Usage:
        motors = MotorInterface()
        motors.move_forward(duration=2)
        motors.turn_left()
        motors.stop()
    """
    
    def __init__(self, ws_url: str = WS_URL):
        self.ws_url = ws_url
        self.ws = None
        self.connected = False
        self.authenticated = False
        self._lock = threading.Lock()
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            self.ws = ws_connect(self.ws_url, open_timeout=5)
            
            # Read welcome message
            welcome = json.loads(self.ws.recv())
            if welcome.get("type") == "welcome":
                print(f"[Motor] ✅ Connected: {welcome.get('message')}")
                self.connected = True
                
                # Authenticate
                return self._authenticate()
            else:
                print(f"[Motor] ⚠️  Unexpected welcome: {welcome}")
                return False
                
        except Exception as e:
            print(f"[Motor] ⚠️  Connection failed: {e}")
            self.connected = False
            return False
    
    def _authenticate(self) -> bool:
        """Authenticate with the server."""
        try:
            self.ws.send(json.dumps({"cmd": "auth", "token": AUTH_TOKEN}))
            response = json.loads(self.ws.recv())
            
            if response.get("success"):
                self.authenticated = True
                print(f"[Motor] ✅ Authenticated")
                return True
            else:
                print(f"[Motor] ❌ Auth failed: {response.get('message')}")
                return False
                
        except Exception as e:
            print(f"[Motor] ❌ Auth error: {e}")
            return False
    
    def _reconnect(self) -> bool:
        """Reconnect if disconnected."""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        return self._connect()
    
    def _send_command(self, cmd: dict) -> dict:
        """Send a command to the server."""
        stats.commands_sent += 1
        
        with self._lock:
            if not self.connected or not self.authenticated:
                if not self._reconnect():
                    stats.failed += 1
                    return {"success": False, "error": "Not connected"}
            
            try:
                self.ws.send(json.dumps(cmd))
                response = json.loads(self.ws.recv())
                
                if response.get("type") == "error":
                    stats.failed += 1
                    return {"success": False, "error": response.get("message")}
                
                stats.successful += 1
                return response
                
            except Exception as e:
                stats.failed += 1
                self.connected = False
                return {"success": False, "error": str(e)}
    
    def _send_move(self, direction: str) -> dict:
        """Send a move command."""
        return self._send_command({"cmd": "move", "dir": direction})
    
    def move_forward(self, duration: float = DEFAULT_DURATION, 
                     speed: float = DEFAULT_SPEED, blocking: bool = True) -> bool:
        """
        Move forward for a specified duration.
        
        Args:
            duration: Time to move in seconds
            speed: Speed (not used by current server)
            blocking: Wait for movement to complete
            
        Returns:
            True if successful
        """
        print(f"[Motor] ⬆️  Forward: {duration}s")
        result = self._send_move("forward")
        
        if result.get("success") and duration > 0:
            stats.total_move_time += duration
            
            if blocking:
                time.sleep(duration)
                self.stop()
            else:
                threading.Timer(duration, self.stop).start()
        
        return result.get("success", False)
    
    def move_backward(self, duration: float = DEFAULT_DURATION,
                      speed: float = DEFAULT_SPEED, blocking: bool = True) -> bool:
        """Move backward for a specified duration."""
        print(f"[Motor] ⬇️  Backward: {duration}s")
        result = self._send_move("backward")
        
        if result.get("success") and duration > 0:
            stats.total_move_time += duration
            
            if blocking:
                time.sleep(duration)
                self.stop()
            else:
                threading.Timer(duration, self.stop).start()
        
        return result.get("success", False)
    
    def turn_left(self, duration: float = TURN_DURATION,
                  speed: float = DEFAULT_SPEED, blocking: bool = True) -> bool:
        """Turn left (approximately 90°)."""
        print(f"[Motor] ⬅️  Turn left: {duration}s")
        result = self._send_move("left")
        
        if result.get("success") and duration > 0:
            stats.total_move_time += duration
            
            if blocking:
                time.sleep(duration)
                self.stop()
            else:
                threading.Timer(duration, self.stop).start()
        
        return result.get("success", False)
    
    def turn_right(self, duration: float = TURN_DURATION,
                   speed: float = DEFAULT_SPEED, blocking: bool = True) -> bool:
        """Turn right (approximately 90°)."""
        print(f"[Motor] ➡️  Turn right: {duration}s")
        result = self._send_move("right")
        
        if result.get("success") and duration > 0:
            stats.total_move_time += duration
            
            if blocking:
                time.sleep(duration)
                self.stop()
            else:
                threading.Timer(duration, self.stop).start()
        
        return result.get("success", False)
    
    def rotate_180(self, blocking: bool = True) -> bool:
        """Rotate 180 degrees."""
        print(f"[Motor] 🔄 Rotate 180°")
        # Turn left for twice the normal duration
        return self.turn_left(duration=ROTATE_180_DURATION, blocking=blocking)
    
    def stop(self) -> bool:
        """Emergency stop."""
        print("[Motor] 🛑 Stop")
        result = self._send_move("stop")
        return result.get("success", False)
    
    def get_status(self) -> Optional[dict]:
        """Get current robot status."""
        result = self._send_command({"cmd": "status"})
        if result.get("type") == "status":
            return result
        return None
    
    def ping(self) -> bool:
        """Ping the server."""
        result = self._send_command({"cmd": "ping"})
        return result.get("type") == "pong"
    
    def close(self):
        """Close the connection."""
        if self.ws:
            try:
                self.stop()
                self.ws.close()
            except:
                pass
            self.ws = None
            self.connected = False
        print("[Motor] 🔌 Disconnected")
    
    def get_stats(self) -> dict:
        """Get motor control statistics."""
        return {
            "commands_sent": stats.commands_sent,
            "successful": stats.successful,
            "failed": stats.failed,
            "total_move_time": round(stats.total_move_time, 1),
            "connected": self.connected,
            "authenticated": self.authenticated,
            "ws_url": self.ws_url
        }


# Singleton instance
_motors: Optional[MotorInterface] = None


def get_motors() -> MotorInterface:
    """Get or create the global motor interface."""
    global _motors
    if _motors is None:
        _motors = MotorInterface()
    return _motors


# =============================================================================
# CLI Testing
# =============================================================================

def main():
    """Test motor control from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Motor Interface for Robot Pet")
    parser.add_argument("--command", "-c", 
                       choices=["forward", "backward", "left", "right", "stop", "rotate", "ping"],
                       help="Movement command")
    parser.add_argument("--duration", "-d", type=float, default=2.0,
                       help="Duration in seconds")
    parser.add_argument("--status", action="store_true", help="Get robot status")
    parser.add_argument("--stats", action="store_true", help="Get control stats")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive control mode")
    parser.add_argument("--server", default=WS_URL, help="WebSocket URL")
    
    args = parser.parse_args()
    
    motors = MotorInterface(args.server)
    
    if args.status:
        status = motors.get_status()
        print("\n📊 Robot Status:")
        print(json.dumps(status, indent=2))
        motors.close()
        return
    
    if args.stats:
        print("\n📊 Motor Stats:")
        print(json.dumps(motors.get_stats(), indent=2))
        motors.close()
        return
    
    if args.command:
        print(f"\n🤖 Executing: {args.command}")
        
        if args.command == "forward":
            motors.move_forward(args.duration)
        elif args.command == "backward":
            motors.move_backward(args.duration)
        elif args.command == "left":
            motors.turn_left(args.duration)
        elif args.command == "right":
            motors.turn_right(args.duration)
        elif args.command == "rotate":
            motors.rotate_180()
        elif args.command == "stop":
            motors.stop()
        elif args.command == "ping":
            print("Pong!" if motors.ping() else "No response")
        
        print(f"📊 Stats: {motors.get_stats()}")
        motors.close()
        return
    
    if args.interactive:
        print("\n" + "="*60)
        print("🕹️  Interactive Motor Control")
        print("="*60)
        print("Commands:")
        print("  w/s - Forward/Backward")
        print("  a/d - Turn Left/Right")
        print("  r   - Rotate 180°")
        print("  x   - Stop")
        print("  q   - Quit")
        print("="*60)
        
        import sys
        import tty
        import termios
        
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            while True:
                char = sys.stdin.read(1).lower()
                
                if char == 'w':
                    motors.move_forward(1.0, blocking=False)
                elif char == 's':
                    motors.move_backward(1.0, blocking=False)
                elif char == 'a':
                    motors.turn_left(0.5, blocking=False)
                elif char == 'd':
                    motors.turn_right(0.5, blocking=False)
                elif char == 'r':
                    motors.rotate_180(blocking=False)
                elif char == 'x':
                    motors.stop()
                elif char == 'q':
                    motors.stop()
                    break
                    
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        print(f"\n📊 Final stats: {motors.get_stats()}")
        motors.close()
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
