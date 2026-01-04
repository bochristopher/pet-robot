#!/usr/bin/env python3
"""
Safety Monitor - Multi-threaded continuous obstacle monitoring
Runs at 50+ Hz to detect obstacles and trigger emergency stops.

Uses ultrasonic as PRIMARY sensor (fast, reliable for close range).
Accepts LiDAR scan data from main loop via update_lidar_scan().
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Callable, List, Tuple
import serial


class ThreatLevel(Enum):
    """Threat levels based on distance."""
    CRITICAL = 1   # <15cm - IMMEDIATE STOP
    DANGER = 2     # <25cm - abort movement
    WARNING = 3    # <40cm - slow down
    CLEAR = 4      # >40cm - safe


class Direction(Enum):
    """Directions around the robot."""
    FRONT = 0
    FRONT_LEFT = 1
    FRONT_RIGHT = 2
    LEFT = 3
    RIGHT = 4
    BACK = 5


@dataclass
class ThreatInfo:
    """Information about detected threat."""
    level: ThreatLevel
    direction: Direction
    distance_cm: float
    source: str  # 'lidar', 'ultrasonic', 'camera'


# Distance thresholds in centimeters
# Larger safety margins - stop further from obstacles
CRITICAL_DISTANCE = 25   # <25cm - IMMEDIATE STOP
DANGER_DISTANCE = 40     # <40cm - abort movement
WARNING_DISTANCE = 60    # <60cm - slow down

# Minimum valid distance (ignore readings below this - likely sensor noise)
MIN_VALID_DISTANCE = 3

# Sensor baseline offsets - subtract from raw readings if sensor sees robot body
# Set to 0 for sensors with clear line of sight
ULTRASONIC_BASELINE = {
    'front_left': 0,
    'front_right': 0,   # Was incorrectly set to 16 - that was a real obstacle!
    'back': 0,
}


class SafetyMonitor:
    """
    Continuous safety monitoring system.

    Runs in a dedicated thread checking ultrasonic sensors at 50+ Hz.
    LiDAR data is fed from the main loop via update_lidar_scan().
    """

    def __init__(self, arduino: serial.Serial,
                 ultrasonic=None,
                 camera=None):
        """
        Initialize safety monitor.

        Args:
            arduino: Serial connection to Arduino for motor control
            ultrasonic: UltrasonicSensors instance (optional but recommended)
            camera: CameraObstacleDetector instance (optional)
        """
        self.arduino = arduino
        self.ultrasonic = ultrasonic
        self.camera = camera

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Current threat status
        self._threat_level = ThreatLevel.CLEAR
        self._threats: Dict[Direction, ThreatInfo] = {}
        self._last_update = 0

        # LiDAR scan data (fed from main loop)
        self._lidar_front_cm = float('inf')
        self._lidar_left_cm = float('inf')
        self._lidar_right_cm = float('inf')
        self._lidar_update_time = 0

        # Callbacks
        self._on_emergency_stop: Optional[Callable] = None
        self._emergency_triggered = False

        # Recovery mode - temporarily disable emergency stops for escape maneuvers
        self._recovery_mode = False
        self._recovery_until = 0

        # Stats for monitoring
        self._loop_count = 0
        self._loop_time_sum = 0
        self._us_read_count = 0

    def set_emergency_callback(self, callback: Callable):
        """Set callback for emergency stop events."""
        self._on_emergency_stop = callback

    def update_lidar_scan(self, front_cm: float, left_cm: float, right_cm: float):
        """
        Update LiDAR distances from main loop.
        Call this whenever you have new LiDAR data.

        Args:
            front_cm: Minimum distance in front (cm)
            left_cm: Minimum distance to left (cm)
            right_cm: Minimum distance to right (cm)
        """
        with self._lock:
            self._lidar_front_cm = front_cm
            self._lidar_left_cm = left_cm
            self._lidar_right_cm = right_cm
            self._lidar_update_time = time.time()

    def start(self):
        """Start the safety monitoring thread."""
        if self._running:
            return

        self._running = True
        self._emergency_triggered = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[Safety] Monitor started (ultrasonic @ 50+ Hz)")

    def stop(self):
        """Stop the safety monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        print("[Safety] Monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop - runs at 50+ Hz."""
        target_interval = 0.02  # 50 Hz = 20ms

        while self._running:
            loop_start = time.time()

            try:
                self._check_all_sensors()
            except Exception as e:
                pass  # Don't let sensor errors stop monitoring

            # Maintain target frequency
            elapsed = time.time() - loop_start
            self._loop_time_sum += elapsed
            self._loop_count += 1

            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _check_all_sensors(self):
        """Check all available sensors and update threat status."""
        threats = {}

        # Check ultrasonic (PRIMARY - fast and accurate for close range)
        if self.ultrasonic:
            us_threats = self._check_ultrasonic()
            threats.update(us_threats)

        # Check LiDAR data (fed from main loop)
        lidar_threats = self._check_lidar_data()
        for direction, threat in lidar_threats.items():
            if direction not in threats or threat.distance_cm < threats[direction].distance_cm:
                threats[direction] = threat

        # Check camera (less frequent - every 5 loops)
        if self.camera and self._loop_count % 5 == 0:
            cam_threats = self._check_camera()
            for direction, threat in cam_threats.items():
                if direction not in threats:
                    threats[direction] = threat

        # Update state atomically
        with self._lock:
            self._threats = threats
            self._last_update = time.time()

            # Determine overall threat level
            if threats:
                worst = min(threats.values(), key=lambda t: t.level.value)
                self._threat_level = worst.level
            else:
                self._threat_level = ThreatLevel.CLEAR

            # Only trigger emergency stop for FRONT obstacles (not side/back)
            front_directions = {Direction.FRONT, Direction.FRONT_LEFT, Direction.FRONT_RIGHT}
            front_critical = any(
                t.level == ThreatLevel.CRITICAL and t.direction in front_directions
                for t in threats.values()
            )
            if front_critical and not self._emergency_triggered:
                self._do_emergency_stop()

    def _check_ultrasonic(self) -> Dict[Direction, ThreatInfo]:
        """Check ultrasonic sensors for obstacles."""
        threats = {}

        try:
            reading = self.ultrasonic.read_all()
            self._us_read_count += 1

            # Front left - apply baseline calibration
            baseline_fl = ULTRASONIC_BASELINE.get('front_left', 0)
            if reading.front_left > baseline_fl and reading.front_left < 400:
                dist_cm = reading.front_left - baseline_fl
                if dist_cm > MIN_VALID_DISTANCE:
                    level = self._distance_to_threat(dist_cm)
                    if level != ThreatLevel.CLEAR:
                        threats[Direction.FRONT_LEFT] = ThreatInfo(
                            level=level,
                            direction=Direction.FRONT_LEFT,
                            distance_cm=dist_cm,
                            source='ultrasonic'
                        )
                    # Also count as FRONT threat if close enough
                    if level in (ThreatLevel.CRITICAL, ThreatLevel.DANGER):
                        if Direction.FRONT not in threats or dist_cm < threats[Direction.FRONT].distance_cm:
                            threats[Direction.FRONT] = ThreatInfo(
                                level=level,
                                direction=Direction.FRONT,
                                distance_cm=dist_cm,
                                source='ultrasonic'
                            )

            # Front right - apply baseline calibration (this sensor sees robot body at ~16cm)
            baseline_fr = ULTRASONIC_BASELINE.get('front_right', 0)
            if reading.front_right > baseline_fr and reading.front_right < 400:
                dist_cm = reading.front_right - baseline_fr
                if dist_cm > MIN_VALID_DISTANCE:
                    level = self._distance_to_threat(dist_cm)
                    if level != ThreatLevel.CLEAR:
                        threats[Direction.FRONT_RIGHT] = ThreatInfo(
                            level=level,
                            direction=Direction.FRONT_RIGHT,
                            distance_cm=dist_cm,
                            source='ultrasonic'
                        )
                    if level in (ThreatLevel.CRITICAL, ThreatLevel.DANGER):
                        if Direction.FRONT not in threats or dist_cm < threats[Direction.FRONT].distance_cm:
                            threats[Direction.FRONT] = ThreatInfo(
                                level=level,
                                direction=Direction.FRONT,
                                distance_cm=dist_cm,
                                source='ultrasonic'
                            )

            # Back - apply baseline calibration
            baseline_back = ULTRASONIC_BASELINE.get('back', 0)
            if reading.back > baseline_back and reading.back < 400:
                dist_cm = reading.back - baseline_back
                if dist_cm > MIN_VALID_DISTANCE:
                    level = self._distance_to_threat(dist_cm)
                    if level != ThreatLevel.CLEAR:
                        threats[Direction.BACK] = ThreatInfo(
                            level=level,
                            direction=Direction.BACK,
                            distance_cm=dist_cm,
                            source='ultrasonic'
                        )

        except Exception as e:
            pass

        return threats

    def _check_lidar_data(self) -> Dict[Direction, ThreatInfo]:
        """Check LiDAR data that was fed from main loop."""
        threats = {}

        with self._lock:
            # Only use LiDAR data if it's fresh (< 200ms old)
            if time.time() - self._lidar_update_time > 0.2:
                return threats

            # Front
            if self._lidar_front_cm < float('inf'):
                level = self._distance_to_threat(self._lidar_front_cm)
                if level != ThreatLevel.CLEAR:
                    threats[Direction.FRONT] = ThreatInfo(
                        level=level,
                        direction=Direction.FRONT,
                        distance_cm=self._lidar_front_cm,
                        source='lidar'
                    )

            # Left
            if self._lidar_left_cm < float('inf'):
                level = self._distance_to_threat(self._lidar_left_cm)
                if level != ThreatLevel.CLEAR:
                    threats[Direction.LEFT] = ThreatInfo(
                        level=level,
                        direction=Direction.LEFT,
                        distance_cm=self._lidar_left_cm,
                        source='lidar'
                    )

            # Right
            if self._lidar_right_cm < float('inf'):
                level = self._distance_to_threat(self._lidar_right_cm)
                if level != ThreatLevel.CLEAR:
                    threats[Direction.RIGHT] = ThreatInfo(
                        level=level,
                        direction=Direction.RIGHT,
                        distance_cm=self._lidar_right_cm,
                        source='lidar'
                    )

        return threats

    def _check_camera(self) -> Dict[Direction, ThreatInfo]:
        """Check camera for obstacles."""
        threats = {}

        try:
            result = self.camera.detect_obstacles()
            if not result:
                return threats

            dist = result.get('obstacle_distance', 'clear')

            # Map camera distance to threat level
            if dist == 'close':
                level = ThreatLevel.CRITICAL
                dist_cm = 10
            elif dist == 'medium':
                level = ThreatLevel.DANGER
                dist_cm = 20
            elif dist == 'far':
                level = ThreatLevel.WARNING
                dist_cm = 35
            else:
                return threats

            # Check which sectors are blocked
            if result.get('left_blocked'):
                threats[Direction.FRONT_LEFT] = ThreatInfo(
                    level=level,
                    direction=Direction.FRONT_LEFT,
                    distance_cm=dist_cm,
                    source='camera'
                )
            if result.get('center_blocked'):
                threats[Direction.FRONT] = ThreatInfo(
                    level=level,
                    direction=Direction.FRONT,
                    distance_cm=dist_cm,
                    source='camera'
                )
            if result.get('right_blocked'):
                threats[Direction.FRONT_RIGHT] = ThreatInfo(
                    level=level,
                    direction=Direction.FRONT_RIGHT,
                    distance_cm=dist_cm,
                    source='camera'
                )

        except Exception as e:
            pass

        return threats

    def _distance_to_threat(self, dist_cm: float) -> ThreatLevel:
        """Convert distance to threat level."""
        if dist_cm < CRITICAL_DISTANCE:
            return ThreatLevel.CRITICAL
        elif dist_cm < DANGER_DISTANCE:
            return ThreatLevel.DANGER
        elif dist_cm < WARNING_DISTANCE:
            return ThreatLevel.WARNING
        else:
            return ThreatLevel.CLEAR

    def _do_emergency_stop(self):
        """Execute emergency stop (unless in recovery mode)."""
        # Check if in recovery mode
        if self._recovery_mode:
            if time.time() < self._recovery_until:
                return  # Skip emergency stop during recovery
            else:
                self._recovery_mode = False  # Recovery period ended

        self._emergency_triggered = True
        try:
            self.arduino.write(b"STOP\n")
            self.arduino.flush()
        except:
            pass

        print("\n[Safety] *** EMERGENCY STOP ***")

        if self._on_emergency_stop:
            try:
                self._on_emergency_stop()
            except:
                pass

    def emergency_stop(self):
        """Manually trigger emergency stop (thread-safe)."""
        with self._lock:
            self._do_emergency_stop()

    def clear_emergency(self):
        """Clear emergency state to allow movement again."""
        with self._lock:
            self._emergency_triggered = False

    def enter_recovery_mode(self, duration: float = 1.0):
        """
        Enter recovery mode - temporarily disable emergency stops.
        Use this when executing escape maneuvers that need to complete.

        Args:
            duration: How long to stay in recovery mode (seconds)
        """
        with self._lock:
            self._recovery_mode = True
            self._recovery_until = time.time() + duration
            self._emergency_triggered = False

    def exit_recovery_mode(self):
        """Exit recovery mode, re-enabling emergency stops."""
        with self._lock:
            self._recovery_mode = False

    def get_threat_level(self) -> ThreatLevel:
        """Get current overall threat level (thread-safe)."""
        with self._lock:
            return self._threat_level

    def get_threats(self) -> Dict[Direction, ThreatInfo]:
        """Get all current threats (thread-safe)."""
        with self._lock:
            return self._threats.copy()

    def get_safe_directions(self) -> Dict[Direction, bool]:
        """Get which directions are safe to move (thread-safe)."""
        with self._lock:
            result = {d: True for d in Direction}
            for direction, threat in self._threats.items():
                if threat.level in (ThreatLevel.CRITICAL, ThreatLevel.DANGER):
                    result[direction] = False
            return result

    def is_safe_to_move(self, direction: Direction) -> bool:
        """Check if it's safe to move in a direction (thread-safe)."""
        with self._lock:
            if direction in self._threats:
                threat = self._threats[direction]
                return threat.level not in (ThreatLevel.CRITICAL, ThreatLevel.DANGER)
            return True

    def is_front_clear(self) -> bool:
        """Check if front is clear for forward movement."""
        safe = self.get_safe_directions()
        return (safe.get(Direction.FRONT, True) and
                safe.get(Direction.FRONT_LEFT, True) and
                safe.get(Direction.FRONT_RIGHT, True))

    def is_back_clear(self) -> bool:
        """Check if back is clear for backward movement."""
        return self.is_safe_to_move(Direction.BACK)

    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        avg_loop = (self._loop_time_sum / self._loop_count * 1000) if self._loop_count > 0 else 0
        return {
            'loop_count': self._loop_count,
            'avg_loop_ms': avg_loop,
            'frequency_hz': 1000 / avg_loop if avg_loop > 0 else 0,
            'emergency_triggered': self._emergency_triggered,
            'ultrasonic_reads': self._us_read_count,
        }


def test_safety_monitor():
    """Test the safety monitor with just ultrasonic."""
    print("=" * 50)
    print("SAFETY MONITOR TEST (Ultrasonic Primary)")
    print("=" * 50)

    # Connect to Arduino
    print("\n[1/3] Arduino...")
    arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2.5)
    arduino.reset_input_buffer()
    arduino.write(b"STOP\n")
    print("  OK")

    # Get ultrasonic
    ultrasonic = None
    print("\n[2/3] Ultrasonic...")
    try:
        from sensors.ultrasonic import UltrasonicSensors
        ultrasonic = UltrasonicSensors(serial_conn=arduino)
        print("  OK")
    except Exception as e:
        print(f"  Failed: {e}")

    # Get camera
    camera = None
    print("\n[3/3] Camera...")
    try:
        from sensors.camera_obstacle import CameraObstacleDetector
        camera = CameraObstacleDetector()
        if camera.open():
            print("  OK")
        else:
            camera = None
    except Exception as e:
        print(f"  Failed: {e}")

    if not ultrasonic:
        print("\nERROR: Ultrasonic required for safety monitor!")
        arduino.close()
        return

    # Create safety monitor
    print("\n" + "=" * 50)
    print("Starting Safety Monitor...")
    print("Put your hand in front of robot to test!")

    monitor = SafetyMonitor(
        arduino=arduino,
        ultrasonic=ultrasonic,
        camera=camera
    )

    def on_emergency():
        print("\n  !!! EMERGENCY CALLBACK !!!")

    monitor.set_emergency_callback(on_emergency)
    monitor.start()

    print("Monitoring for 15 seconds (Ctrl+C to stop)...")
    print("-" * 50)

    try:
        for i in range(150):
            time.sleep(0.1)

            level = monitor.get_threat_level()
            threats = monitor.get_threats()

            # Color coding
            if level == ThreatLevel.CRITICAL:
                level_str = f"\033[91mCRITICAL\033[0m"
            elif level == ThreatLevel.DANGER:
                level_str = f"\033[93mDANGER\033[0m"
            elif level == ThreatLevel.WARNING:
                level_str = f"\033[94mWARNING\033[0m"
            else:
                level_str = "CLEAR"

            threat_str = ", ".join([f"{t.source[0].upper()}:{t.direction.name}={t.distance_cm:.0f}cm"
                                    for t in threats.values()])
            if not threat_str:
                threat_str = "none"

            print(f"\r[{i:3d}] {level_str:20} | {threat_str:50}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    # Stats
    stats = monitor.get_stats()
    print(f"\n\nStats:")
    print(f"  Loops: {stats['loop_count']}")
    print(f"  Avg loop: {stats['avg_loop_ms']:.1f}ms")
    print(f"  Frequency: {stats['frequency_hz']:.0f} Hz")
    print(f"  US reads: {stats['ultrasonic_reads']}")

    # Cleanup
    monitor.stop()
    if camera:
        camera.close()
    arduino.close()
    print("\nDone!")


if __name__ == "__main__":
    test_safety_monitor()
