#!/usr/bin/env python3
"""
Odometry & Position Tracking - Phase 2 of SLAM System

Provides accurate robot position tracking through:
- Wheel encoder odometry with calibration
- IMU heading integration (gyroscope)
- Simple sensor fusion for robust estimates
- Covariance tracking for uncertainty
"""

import time
import math
import threading
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class Pose:
    """Robot pose with uncertainty."""
    x: float = 0.0          # meters
    y: float = 0.0          # meters
    theta: float = 0.0      # radians

    # Uncertainty (standard deviations)
    x_std: float = 0.01     # meters
    y_std: float = 0.01     # meters
    theta_std: float = 0.02 # radians

    # Timestamp
    timestamp: float = 0.0

    def copy(self) -> 'Pose':
        return Pose(
            self.x, self.y, self.theta,
            self.x_std, self.y_std, self.theta_std,
            self.timestamp
        )


class WheelOdometry:
    """
    Differential drive wheel odometry.

    Converts encoder ticks to position changes.
    Supports runtime calibration.
    """

    def __init__(self,
                 wheel_base: float = 0.20,      # Distance between wheels (m)
                 wheel_diameter: float = 0.065,  # Wheel diameter (m)
                 ticks_per_rev: float = 360,     # Encoder ticks per wheel revolution
                 left_scale: float = 1.0,        # Left wheel scale factor
                 right_scale: float = 1.0):      # Right wheel scale factor
        """
        Initialize wheel odometry.

        Args:
            wheel_base: Distance between wheels in meters
            wheel_diameter: Wheel diameter in meters
            ticks_per_rev: Encoder pulses per wheel revolution
            left_scale: Calibration scale for left wheel
            right_scale: Calibration scale for right wheel
        """
        self.wheel_base = wheel_base
        self.wheel_diameter = wheel_diameter
        self.ticks_per_rev = ticks_per_rev
        self.left_scale = left_scale
        self.right_scale = right_scale

        # Derived
        self.wheel_circumference = math.pi * wheel_diameter
        self.meters_per_tick = self.wheel_circumference / ticks_per_rev

        # Last encoder readings
        self._last_left = 0
        self._last_right = 0
        self._initialized = False

    def set_calibration(self, left_scale: float, right_scale: float):
        """Update wheel calibration factors."""
        self.left_scale = left_scale
        self.right_scale = right_scale

    def update(self, left_ticks: int, right_ticks: int) -> Tuple[float, float, float]:
        """
        Compute position change from encoder readings.

        Args:
            left_ticks: Current left encoder count
            right_ticks: Current right encoder count

        Returns:
            (dx, dy, dtheta) position change in robot frame
        """
        if not self._initialized:
            self._last_left = left_ticks
            self._last_right = right_ticks
            self._initialized = True
            return 0.0, 0.0, 0.0

        # Compute tick deltas
        d_left_ticks = left_ticks - self._last_left
        d_right_ticks = right_ticks - self._last_right

        self._last_left = left_ticks
        self._last_right = right_ticks

        # Convert to distances with calibration
        d_left = d_left_ticks * self.meters_per_tick * self.left_scale
        d_right = d_right_ticks * self.meters_per_tick * self.right_scale

        # Differential drive kinematics
        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.wheel_base

        # Motion in robot frame
        if abs(d_theta) < 1e-6:
            # Straight line motion
            dx = d_center
            dy = 0.0
        else:
            # Arc motion
            radius = d_center / d_theta
            dx = radius * math.sin(d_theta)
            dy = radius * (1 - math.cos(d_theta))

        return dx, dy, d_theta

    def reset(self, left_ticks: int = 0, right_ticks: int = 0):
        """Reset encoder baseline."""
        self._last_left = left_ticks
        self._last_right = right_ticks
        self._initialized = True


class IMUHeading:
    """
    IMU-based heading tracking.

    Uses gyroscope for angular velocity integration.
    More accurate than wheel-based heading for turning.
    """

    def __init__(self, imu, drift_rate: float = 0.0):
        """
        Initialize IMU heading tracker.

        Args:
            imu: IMU object with get_heading() method (degrees)
            drift_rate: Known gyro drift in rad/s (for compensation)
        """
        self.imu = imu
        self.drift_rate = drift_rate

        self._last_time = time.time()
        self._drift_accumulated = 0.0

    def get_heading(self) -> Tuple[float, float]:
        """
        Get current heading.

        Returns:
            (heading_rad, std_dev) heading and uncertainty
        """
        try:
            heading_deg = self.imu.get_heading()
            heading_rad = math.radians(heading_deg)

            # Compensate for accumulated drift
            now = time.time()
            dt = now - self._last_time
            self._drift_accumulated += self.drift_rate * dt
            self._last_time = now

            heading_rad -= self._drift_accumulated

            # Normalize to [-pi, pi]
            while heading_rad > math.pi:
                heading_rad -= 2 * math.pi
            while heading_rad < -math.pi:
                heading_rad += 2 * math.pi

            # Uncertainty increases with time due to drift
            # Typical MEMS gyro: 0.01 rad/s drift
            std_dev = 0.02 + 0.01 * dt

            return heading_rad, std_dev

        except Exception:
            return None, None

    def reset(self):
        """Reset heading to zero."""
        try:
            self.imu.reset_yaw()
            self._drift_accumulated = 0.0
            self._last_time = time.time()
        except:
            pass


class Odometry:
    """
    Full odometry system with sensor fusion.

    Combines wheel encoders and IMU for robust position tracking.
    Runs in background thread for continuous updates.
    """

    def __init__(self,
                 arduino,
                 imu=None,
                 wheel_base: float = 0.20,
                 wheel_diameter: float = 0.065,
                 ticks_per_rev: float = 360,
                 update_rate: float = 50.0):
        """
        Initialize odometry system.

        Args:
            arduino: Serial connection to Arduino
            imu: Optional IMU object for heading
            wheel_base: Distance between wheels (m)
            wheel_diameter: Wheel diameter (m)
            ticks_per_rev: Encoder ticks per revolution
            update_rate: Update frequency (Hz)
        """
        self.arduino = arduino

        # Wheel odometry
        self.wheel_odom = WheelOdometry(
            wheel_base=wheel_base,
            wheel_diameter=wheel_diameter,
            ticks_per_rev=ticks_per_rev
        )

        # IMU heading (if available)
        self.imu_heading = IMUHeading(imu) if imu else None

        # Current pose
        self._pose = Pose(timestamp=time.time())
        self._pose_lock = threading.RLock()

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._update_interval = 1.0 / update_rate

        # Callbacks
        self._on_pose_update: Optional[Callable[[Pose], None]] = None

        # Statistics
        self._updates = 0
        self._total_distance = 0.0
        self._total_rotation = 0.0

    def set_pose_callback(self, callback: Callable[[Pose], None]):
        """Set callback for pose updates."""
        self._on_pose_update = callback

    def get_pose(self) -> Pose:
        """Get current pose (thread-safe copy)."""
        with self._pose_lock:
            return self._pose.copy()

    def set_pose(self, x: float, y: float, theta: float):
        """Set current pose (for reset or correction)."""
        with self._pose_lock:
            self._pose.x = x
            self._pose.y = y
            self._pose.theta = theta
            self._pose.timestamp = time.time()

    def reset(self):
        """Reset to origin."""
        with self._pose_lock:
            self._pose = Pose(timestamp=time.time())

        # Reset IMU heading
        if self.imu_heading:
            self.imu_heading.reset()

        # Reset encoder baseline
        left, right = self._get_encoders()
        self.wheel_odom.reset(left, right)

        self._total_distance = 0.0
        self._total_rotation = 0.0

    def _get_encoders(self) -> Tuple[int, int]:
        """Read encoder values from Arduino."""
        try:
            self.arduino.write(b"ENCODERS\n")
            self.arduino.flush()
            time.sleep(0.01)

            while self.arduino.in_waiting:
                resp = self.arduino.readline().decode().strip()
                if resp.startswith("ENC:"):
                    parts = resp[4:].split(",")
                    return int(parts[0]), int(parts[1])
        except:
            pass
        return 0, 0

    def update(self):
        """
        Single odometry update.

        Call this in main loop if not using background thread.
        """
        # Get encoder readings
        left, right = self._get_encoders()

        # Compute motion from wheels
        dx, dy, dtheta_wheels = self.wheel_odom.update(left, right)

        with self._pose_lock:
            # Get heading from IMU if available (more accurate for rotation)
            if self.imu_heading:
                imu_heading, imu_std = self.imu_heading.get_heading()
                if imu_heading is not None:
                    # Use IMU heading directly (it's absolute)
                    self._pose.theta = imu_heading
                    self._pose.theta_std = imu_std
                else:
                    # Fall back to wheel-based heading
                    self._pose.theta += dtheta_wheels
                    self._pose.theta_std += 0.01 * abs(dtheta_wheels)
            else:
                # Pure wheel-based heading
                self._pose.theta += dtheta_wheels
                self._pose.theta_std += 0.01 * abs(dtheta_wheels)

            # Normalize heading
            while self._pose.theta > math.pi:
                self._pose.theta -= 2 * math.pi
            while self._pose.theta < -math.pi:
                self._pose.theta += 2 * math.pi

            # Update position in world frame
            cos_theta = math.cos(self._pose.theta)
            sin_theta = math.sin(self._pose.theta)

            self._pose.x += dx * cos_theta - dy * sin_theta
            self._pose.y += dx * sin_theta + dy * cos_theta

            # Update uncertainty (grows with motion)
            distance = math.sqrt(dx*dx + dy*dy)
            self._pose.x_std += 0.02 * distance  # 2% of distance
            self._pose.y_std += 0.02 * distance

            self._pose.timestamp = time.time()

            # Statistics
            self._updates += 1
            self._total_distance += distance
            self._total_rotation += abs(dtheta_wheels)

        # Callback
        if self._on_pose_update:
            try:
                self._on_pose_update(self._pose)
            except:
                pass

    def start(self):
        """Start background odometry thread."""
        if self._running:
            return

        # Initialize encoder baseline
        left, right = self._get_encoders()
        self.wheel_odom.reset(left, right)

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        print("[Odometry] Started background thread")

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        print("[Odometry] Stopped")

    def _update_loop(self):
        """Background update loop."""
        while self._running:
            try:
                self.update()
            except Exception as e:
                pass
            time.sleep(self._update_interval)

    def get_stats(self) -> dict:
        """Get odometry statistics."""
        pose = self.get_pose()
        return {
            'x': pose.x,
            'y': pose.y,
            'theta_deg': math.degrees(pose.theta),
            'x_std': pose.x_std,
            'y_std': pose.y_std,
            'theta_std_deg': math.degrees(pose.theta_std),
            'updates': self._updates,
            'total_distance_m': self._total_distance,
            'total_rotation_deg': math.degrees(self._total_rotation),
            'has_imu': self.imu_heading is not None,
        }

    def calibrate_straight_line(self, actual_distance: float,
                                 left_ticks: int, right_ticks: int):
        """
        Calibrate wheel scale factors from straight-line test.

        Drive robot in straight line for known distance,
        then call this with actual distance and encoder readings.
        """
        measured_left = left_ticks * self.wheel_odom.meters_per_tick
        measured_right = right_ticks * self.wheel_odom.meters_per_tick

        if measured_left > 0:
            self.wheel_odom.left_scale = actual_distance / measured_left
        if measured_right > 0:
            self.wheel_odom.right_scale = actual_distance / measured_right

        print(f"[Odometry] Calibrated: left_scale={self.wheel_odom.left_scale:.3f}, "
              f"right_scale={self.wheel_odom.right_scale:.3f}")


def test_odometry():
    """Test odometry with simulated data."""
    print("=" * 60)
    print("ODOMETRY TEST (SIMULATED)")
    print("=" * 60)

    # Create mock Arduino
    class MockArduino:
        def __init__(self):
            self.left = 0
            self.right = 0
            self.in_waiting = 0

        def write(self, data):
            # Simulate encoder response
            if b"ENCODERS" in data:
                self.in_waiting = 1

        def flush(self):
            pass

        def readline(self):
            resp = f"ENC:{self.left},{self.right}\n"
            self.in_waiting = 0
            return resp.encode()

        def simulate_forward(self, ticks):
            self.left += ticks
            self.right += ticks

        def simulate_turn_left(self, ticks):
            self.right += ticks

        def simulate_turn_right(self, ticks):
            self.left += ticks

    mock_arduino = MockArduino()

    # Create odometry (no IMU for test)
    odom = Odometry(
        arduino=mock_arduino,
        imu=None,
        wheel_base=0.20,
        wheel_diameter=0.065,
        ticks_per_rev=360
    )

    # Test 1: Forward motion
    print("\n1. Forward motion test...")
    odom.reset()
    mock_arduino.simulate_forward(360)  # One wheel revolution
    odom.update()

    pose = odom.get_pose()
    expected_distance = math.pi * 0.065  # One circumference
    print(f"   Expected: x={expected_distance:.3f}m, y=0, theta=0")
    print(f"   Got:      x={pose.x:.3f}m, y={pose.y:.3f}m, theta={math.degrees(pose.theta):.1f}deg")

    # Test 2: Turn in place
    print("\n2. Turn in place test...")
    odom.reset()
    mock_arduino.left = 0
    mock_arduino.right = 0
    odom.wheel_odom.reset(0, 0)

    # Turn left 90 degrees
    # For 90 deg turn: arc_length = wheel_base/2 * pi/2
    arc_length = 0.20 / 2 * math.pi / 2
    ticks_for_90 = int(arc_length / (math.pi * 0.065) * 360)

    mock_arduino.simulate_turn_left(ticks_for_90)
    odom.update()

    pose = odom.get_pose()
    print(f"   Expected: ~90deg left turn")
    print(f"   Got:      theta={math.degrees(pose.theta):.1f}deg")
    print(f"   Position: x={pose.x:.3f}m, y={pose.y:.3f}m")

    # Test 3: Square path
    print("\n3. Square path test...")
    odom.reset()
    mock_arduino.left = 0
    mock_arduino.right = 0
    odom.wheel_odom.reset(0, 0)

    ticks_per_meter = 360 / (math.pi * 0.065)
    ticks_1m = int(ticks_per_meter)

    positions = []

    # Drive 1m forward
    mock_arduino.simulate_forward(ticks_1m)
    odom.update()
    positions.append(odom.get_pose())

    # Turn left 90
    mock_arduino.simulate_turn_left(ticks_for_90)
    odom.update()

    # Drive 1m forward
    mock_arduino.simulate_forward(ticks_1m)
    odom.update()
    positions.append(odom.get_pose())

    # Turn left 90
    mock_arduino.simulate_turn_left(ticks_for_90)
    odom.update()

    # Drive 1m forward
    mock_arduino.simulate_forward(ticks_1m)
    odom.update()
    positions.append(odom.get_pose())

    # Turn left 90
    mock_arduino.simulate_turn_left(ticks_for_90)
    odom.update()

    # Drive 1m forward (back to start)
    mock_arduino.simulate_forward(ticks_1m)
    odom.update()
    positions.append(odom.get_pose())

    print("   Square corners:")
    for i, p in enumerate(positions):
        print(f"   Corner {i+1}: ({p.x:.2f}, {p.y:.2f}) @ {math.degrees(p.theta):.0f}deg")

    final = positions[-1]
    print(f"   Closure error: {math.sqrt(final.x**2 + final.y**2):.3f}m")

    # Stats
    print("\n4. Statistics:")
    stats = odom.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("Odometry test complete!")


def test_with_hardware():
    """Test odometry with real hardware."""
    print("=" * 60)
    print("ODOMETRY TEST (HARDWARE)")
    print("=" * 60)

    import sys
    import serial
    sys.path.insert(0, '/home/bo/robot_pet')

    # Connect to Arduino
    print("\n1. Connecting to Arduino...")
    try:
        arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        time.sleep(2.5)
        arduino.reset_input_buffer()
        arduino.write(b"RESET\n")
        time.sleep(0.1)
        print("   OK")
    except Exception as e:
        print(f"   Failed: {e}")
        return

    # Try to get IMU
    imu = None
    try:
        from imu_mpu6050 import MPU6050
        imu = MPU6050(bus_num=7)
        imu.calibrate(50)
        imu.reset_yaw()
        print("\n2. IMU connected and calibrated")
    except Exception as e:
        print(f"\n2. IMU not available: {e}")

    # Create odometry
    odom = Odometry(
        arduino=arduino,
        imu=imu,
        wheel_base=0.20,
        wheel_diameter=0.065,
        ticks_per_rev=360
    )

    # Start background updates
    odom.start()

    print("\n3. Monitoring position for 10 seconds...")
    print("   Move the robot around manually")
    print("-" * 40)

    try:
        start = time.time()
        while time.time() - start < 10:
            pose = odom.get_pose()
            print(f"\r   Pos: ({pose.x:.2f}, {pose.y:.2f}) @ {math.degrees(pose.theta):.0f}deg  "
                  f"Uncertainty: {pose.x_std:.3f}m, {math.degrees(pose.theta_std):.1f}deg    ", end="")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    print("\n")

    # Final stats
    print("\n4. Final statistics:")
    stats = odom.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    # Cleanup
    odom.stop()
    arduino.close()

    print("\n" + "=" * 60)
    print("Hardware odometry test complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--hardware":
        test_with_hardware()
    else:
        test_odometry()
