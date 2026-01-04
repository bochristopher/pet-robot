#!/usr/bin/env python3
"""
Movement Controller - Interruptible movement execution
Executes movements in small chunks with safety checks between each.
"""

import time
import math
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
import serial

from safety_monitor import SafetyMonitor, ThreatLevel, Direction


class MovementResult(Enum):
    """Result of a movement attempt."""
    SUCCESS = 1          # Movement completed fully
    ABORTED_OBSTACLE = 2 # Stopped due to obstacle
    ABORTED_EMERGENCY = 3  # Emergency stop triggered
    ABORTED_COMMAND = 4  # User/system commanded stop
    FAILED = 5           # Movement failed (stuck, etc.)


@dataclass
class MovementStatus:
    """Detailed status of a movement."""
    result: MovementResult
    distance_moved: float   # Estimated distance traveled
    duration_actual: float  # Actual time moved
    duration_requested: float
    abort_reason: Optional[str] = None


class MovementController:
    """
    Interruptible movement controller with safety integration.

    Executes movements in small increments (default 50ms) and
    checks the SafetyMonitor between each increment.
    """

    def __init__(self, arduino: serial.Serial,
                 safety: SafetyMonitor,
                 check_interval: float = 0.05):
        """
        Initialize movement controller.

        Args:
            arduino: Serial connection to Arduino
            safety: SafetyMonitor instance for safety checks
            check_interval: Time between safety checks (default 50ms)
        """
        self.arduino = arduino
        self.safety = safety
        self.check_interval = check_interval

        # State
        self._current_command: Optional[str] = None
        self._abort_requested = False
        self._lock = threading.Lock()

        # Odometry callback (optional)
        self._get_encoders: Optional[Callable] = None

        # Estimated speeds (meters per second, calibrate these!)
        self.forward_speed = 0.15   # ~15 cm/s
        self.backward_speed = 0.12  # ~12 cm/s
        self.turn_speed = 1.0       # ~1 rad/s

    def set_encoder_callback(self, callback: Callable):
        """Set callback to get encoder values: callback() -> (left, right)."""
        self._get_encoders = callback

    def _send_command(self, cmd: str):
        """Send command to Arduino."""
        with self._lock:
            self._current_command = cmd
            try:
                self.arduino.write(f"{cmd}\n".encode())
                self.arduino.flush()
                time.sleep(0.02)  # Brief delay for Arduino to process
            except:
                pass

    def _stop(self):
        """Stop motors immediately."""
        with self._lock:
            self._current_command = None
            try:
                self.arduino.write(b"STOP\n")
                self.arduino.flush()
            except:
                pass

    def abort(self):
        """Request abort of current movement (thread-safe)."""
        with self._lock:
            self._abort_requested = True

    def forward(self, duration: float, check_interval: float = None) -> MovementStatus:
        """
        Move forward with safety checks.

        Args:
            duration: How long to move (seconds)
            check_interval: Override default check interval

        Returns:
            MovementStatus with result and details
        """
        if check_interval is None:
            check_interval = self.check_interval

        return self._execute_movement(
            command="FORWARD",
            duration=duration,
            check_interval=check_interval,
            safe_check=lambda: self.safety.is_front_clear(),
            speed=self.forward_speed
        )

    def backward(self, duration: float, check_interval: float = None) -> MovementStatus:
        """
        Move backward with safety checks.

        Args:
            duration: How long to move (seconds)
            check_interval: Override default check interval

        Returns:
            MovementStatus with result and details
        """
        if check_interval is None:
            check_interval = self.check_interval

        return self._execute_movement(
            command="BACKWARD",
            duration=duration,
            check_interval=check_interval,
            safe_check=lambda: self.safety.is_back_clear(),
            speed=self.backward_speed
        )

    def turn_left(self, duration: float, check_interval: float = None) -> MovementStatus:
        """
        Turn left with safety checks.

        Args:
            duration: How long to turn (seconds)
            check_interval: Override default check interval

        Returns:
            MovementStatus with result and details
        """
        if check_interval is None:
            check_interval = self.check_interval

        # Turning is generally safe unless we're in a critical situation
        return self._execute_movement(
            command="LEFT",
            duration=duration,
            check_interval=check_interval,
            safe_check=lambda: self.safety.get_threat_level() != ThreatLevel.CRITICAL,
            speed=self.turn_speed
        )

    def turn_right(self, duration: float, check_interval: float = None) -> MovementStatus:
        """
        Turn right with safety checks.

        Args:
            duration: How long to turn (seconds)
            check_interval: Override default check interval

        Returns:
            MovementStatus with result and details
        """
        if check_interval is None:
            check_interval = self.check_interval

        return self._execute_movement(
            command="RIGHT",
            duration=duration,
            check_interval=check_interval,
            safe_check=lambda: self.safety.get_threat_level() != ThreatLevel.CRITICAL,
            speed=self.turn_speed
        )

    def _execute_movement(self, command: str, duration: float,
                          check_interval: float,
                          safe_check: Callable[[], bool],
                          speed: float) -> MovementStatus:
        """
        Execute a movement with interleaved safety checks.

        Args:
            command: Motor command (FORWARD, BACKWARD, LEFT, RIGHT)
            duration: Total movement duration
            check_interval: Time between safety checks
            safe_check: Function returning True if safe to continue
            speed: Estimated speed for distance calculation

        Returns:
            MovementStatus with result and details
        """
        self._abort_requested = False
        start_time = time.time()
        elapsed = 0

        # Check safety BEFORE starting
        if not safe_check():
            self._stop()
            return MovementStatus(
                result=MovementResult.ABORTED_OBSTACLE,
                distance_moved=0,
                duration_actual=0,
                duration_requested=duration,
                abort_reason="Unsafe at start"
            )

        # Start movement
        self._send_command(command)

        # Execute in chunks with safety checks
        while elapsed < duration:
            # Sleep for one interval (or remaining time)
            sleep_time = min(check_interval, duration - elapsed)
            time.sleep(sleep_time)
            elapsed = time.time() - start_time

            # Check for abort request
            if self._abort_requested:
                self._stop()
                return MovementStatus(
                    result=MovementResult.ABORTED_COMMAND,
                    distance_moved=elapsed * speed,
                    duration_actual=elapsed,
                    duration_requested=duration,
                    abort_reason="Abort requested"
                )

            # Check safety
            if not safe_check():
                self._stop()
                return MovementStatus(
                    result=MovementResult.ABORTED_OBSTACLE,
                    distance_moved=elapsed * speed,
                    duration_actual=elapsed,
                    duration_requested=duration,
                    abort_reason="Obstacle detected"
                )

            # Check for emergency stop
            if self.safety.get_threat_level() == ThreatLevel.CRITICAL:
                self._stop()
                return MovementStatus(
                    result=MovementResult.ABORTED_EMERGENCY,
                    distance_moved=elapsed * speed,
                    duration_actual=elapsed,
                    duration_requested=duration,
                    abort_reason="Emergency stop"
                )

        # Movement completed
        self._stop()
        return MovementStatus(
            result=MovementResult.SUCCESS,
            distance_moved=duration * speed,
            duration_actual=elapsed,
            duration_requested=duration
        )

    def turn_to_heading(self, target_theta: float, current_theta: float,
                        tolerance: float = 0.15) -> MovementStatus:
        """
        Turn to face a specific heading.

        Args:
            target_theta: Target heading in radians
            current_theta: Current heading in radians
            tolerance: Acceptable error in radians (~8 degrees default)

        Returns:
            MovementStatus from the turn
        """
        diff = target_theta - current_theta

        # Normalize to -pi to pi
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi

        if abs(diff) < tolerance:
            return MovementStatus(
                result=MovementResult.SUCCESS,
                distance_moved=0,
                duration_actual=0,
                duration_requested=0
            )

        # Proportional turn time
        turn_time = min(abs(diff) * 0.3, 0.5)

        if diff > 0:
            return self.turn_left(turn_time)
        else:
            return self.turn_right(turn_time)

    def navigate_toward(self, target_x: float, target_y: float,
                        current_x: float, current_y: float,
                        current_theta: float) -> MovementStatus:
        """
        Take one navigation step toward a target position.

        Args:
            target_x, target_y: Target position
            current_x, current_y: Current position
            current_theta: Current heading

        Returns:
            MovementStatus from the movement taken
        """
        # Calculate angle to target
        dx = target_x - current_x
        dy = target_y - current_y
        dist = math.sqrt(dx*dx + dy*dy)
        angle_to_target = math.atan2(dy, dx)

        # Calculate heading difference
        angle_diff = angle_to_target - current_theta
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Turn if needed
        if abs(angle_diff) > 0.3:  # >17 degrees
            turn_time = min(abs(angle_diff) * 0.25, 0.4)
            if angle_diff > 0:
                return self.turn_left(turn_time)
            else:
                return self.turn_right(turn_time)

        # Move forward
        move_time = min(dist * 0.8, 0.8)  # Longer for farther goals
        return self.forward(move_time)

    def escape_obstacle(self) -> MovementStatus:
        """
        Emergency escape from obstacle - backup and turn.

        Returns:
            MovementStatus from the escape maneuver
        """
        # Try to backup
        safe = self.safety.get_safe_directions()

        if safe.get(Direction.BACK, False):
            result = self.backward(0.3)
            if result.result == MovementResult.ABORTED_OBSTACLE:
                pass  # Can't backup, try turning

        # Turn away from obstacle
        threats = self.safety.get_threats()

        # Find which side has fewer threats
        left_threats = sum(1 for d in [Direction.FRONT_LEFT, Direction.LEFT]
                           if d in threats)
        right_threats = sum(1 for d in [Direction.FRONT_RIGHT, Direction.RIGHT]
                            if d in threats)

        if left_threats <= right_threats:
            return self.turn_left(0.4)
        else:
            return self.turn_right(0.4)


def test_movement_controller():
    """Test the movement controller."""
    import sys
    sys.path.insert(0, '/home/bo/robot_pet')

    print("=" * 50)
    print("MOVEMENT CONTROLLER TEST")
    print("=" * 50)

    # Connect to Arduino
    print("\n[1/3] Arduino...")
    arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2.5)
    arduino.reset_input_buffer()
    arduino.write(b"STOP\n")
    print("  OK")

    # Create safety monitor (minimal - just for testing)
    print("\n[2/3] Safety Monitor...")

    # Try to get LiDAR
    lidar = None
    try:
        from sensors.lidar_rplidar import LidarRPLidar
        lidar = LidarRPLidar()
        if lidar.connected:
            lidar.start_scanning()
            time.sleep(1)
            print("  LiDAR OK")
    except:
        print("  No LiDAR")

    # Try ultrasonic
    ultrasonic = None
    try:
        from sensors.ultrasonic import UltrasonicSensors
        ultrasonic = UltrasonicSensors(serial_conn=arduino)
        print("  Ultrasonic OK")
    except:
        print("  No ultrasonic")

    safety = SafetyMonitor(
        arduino=arduino,
        lidar=lidar,
        ultrasonic=ultrasonic,
        camera=None
    )
    safety.start()
    time.sleep(0.5)
    print("  Safety Monitor started")

    # Create movement controller
    print("\n[3/3] Movement Controller...")
    controller = MovementController(arduino, safety)
    print("  OK")

    print("\n" + "=" * 50)
    print("Testing movements (with safety)...")
    print("-" * 50)

    # Test forward
    print("\n1. Forward 0.5s...")
    result = controller.forward(0.5)
    print(f"   Result: {result.result.name}")
    print(f"   Duration: {result.duration_actual:.2f}s / {result.duration_requested:.2f}s")
    if result.abort_reason:
        print(f"   Reason: {result.abort_reason}")
    time.sleep(0.5)

    # Test turn left
    print("\n2. Turn left 0.3s...")
    result = controller.turn_left(0.3)
    print(f"   Result: {result.result.name}")
    time.sleep(0.5)

    # Test turn right
    print("\n3. Turn right 0.3s...")
    result = controller.turn_right(0.3)
    print(f"   Result: {result.result.name}")
    time.sleep(0.5)

    # Test backward
    print("\n4. Backward 0.3s...")
    result = controller.backward(0.3)
    print(f"   Result: {result.result.name}")

    # Show safety stats
    stats = safety.get_stats()
    print(f"\nSafety stats: {stats['frequency_hz']:.0f} Hz monitoring")

    # Cleanup
    print("\nCleaning up...")
    safety.stop()
    if lidar:
        lidar.close()
    arduino.close()

    print("Done!")


if __name__ == "__main__":
    test_movement_controller()
