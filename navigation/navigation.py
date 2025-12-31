#!/usr/bin/env python3
"""
Navigation Module
Fuses IMU and encoder data for robot localization.
Uses complementary filter to combine wheel odometry with IMU heading.
"""

import time
import math
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from imu_mpu6050 import MPU6050, IMUData
from encoder_interface import EncoderInterface, EncoderData

# Robot parameters (adjust for your robot)
WHEEL_DIAMETER_MM = 80.0
ENCODER_CPR = 360  # Counts per revolution
WHEEL_BASE_MM = 200.0  # Distance between left and right wheels


@dataclass
class Pose:
    """Robot pose in 2D."""
    x: float = 0.0  # mm
    y: float = 0.0  # mm
    theta: float = 0.0  # radians
    timestamp: float = 0.0


@dataclass
class Velocity:
    """Robot velocity."""
    linear: float = 0.0  # mm/s
    angular: float = 0.0  # rad/s


@dataclass
class NavigationState:
    """Complete navigation state."""
    pose: Pose
    velocity: Velocity
    imu_yaw: float  # IMU heading (degrees)
    encoder_left: int
    encoder_right: int
    fused_heading: float  # Fused heading (degrees)


class Navigation:
    """
    Robot navigation using sensor fusion.

    Combines:
    - Wheel encoders for distance/velocity
    - IMU gyroscope for heading (drift-corrected)
    """

    def __init__(self,
                 wheel_diameter_mm: float = WHEEL_DIAMETER_MM,
                 encoder_cpr: int = ENCODER_CPR,
                 wheel_base_mm: float = WHEEL_BASE_MM,
                 imu_weight: float = 0.7):
        """
        Initialize navigation.

        Args:
            wheel_diameter_mm: Wheel diameter in mm
            encoder_cpr: Encoder counts per revolution
            wheel_base_mm: Distance between wheels
            imu_weight: Weight for IMU in heading fusion (0-1)
        """
        self.wheel_circumference = wheel_diameter_mm * math.pi
        self.mm_per_count = self.wheel_circumference / encoder_cpr
        self.wheel_base = wheel_base_mm
        self.imu_weight = imu_weight

        # Sensors
        self.imu: Optional[MPU6050] = None
        self.encoders: Optional[EncoderInterface] = None

        # State
        self.pose = Pose()
        self.velocity = Velocity()
        self.last_left_count = 0
        self.last_right_count = 0
        self.last_time = None

        # Heading fusion
        self.encoder_heading = 0.0  # From wheel odometry
        self.imu_heading = 0.0  # From IMU
        self.fused_heading = 0.0  # Combined

        # Path history
        self.path: List[Tuple[float, float]] = []
        self.max_path_points = 1000

        # Thread for continuous updates
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self._initialized = False

    def init_sensors(self, calibrate_imu: bool = True, serial_conn=None) -> bool:
        """
        Initialize IMU and encoder sensors.

        Args:
            calibrate_imu: Whether to calibrate IMU on startup
            serial_conn: Optional existing serial connection to share with encoders
        """
        try:
            print("[Nav] Initializing sensors...")

            # Initialize IMU
            self.imu = MPU6050(bus_num=7)
            if calibrate_imu:
                self.imu.calibrate(100)

            # Initialize encoders (can share serial connection)
            self.encoders = EncoderInterface(serial_conn=serial_conn)
            if not self.encoders.connected:
                print("[Nav] Warning: Encoder connection failed")

            self._initialized = True
            print("[Nav] Sensors initialized")
            return True

        except Exception as e:
            print(f"[Nav] Sensor init failed: {e}")
            return False

    def reset(self):
        """Reset navigation state to origin."""
        with self._lock:
            self.pose = Pose()
            self.velocity = Velocity()
            self.encoder_heading = 0.0
            self.fused_heading = 0.0
            self.path.clear()
            self.last_time = None

            if self.imu:
                self.imu.reset_yaw()
            if self.encoders:
                self.encoders.reset_encoders()
                self.last_left_count = 0
                self.last_right_count = 0

        print("[Nav] Navigation reset")

    def update(self) -> NavigationState:
        """Update navigation state from sensors."""
        now = time.time()

        # Read sensors
        imu_data = self.imu.read() if self.imu else None
        enc_data = self.encoders.read_encoders() if self.encoders else None

        with self._lock:
            # Update from encoders
            if enc_data:
                left_delta = enc_data.left_count - self.last_left_count
                right_delta = enc_data.right_count - self.last_right_count

                # Distance traveled by each wheel
                left_dist = left_delta * self.mm_per_count
                right_dist = right_delta * self.mm_per_count

                # Linear and angular displacement
                linear_dist = (left_dist + right_dist) / 2.0
                angular_disp = (right_dist - left_dist) / self.wheel_base

                # Update encoder-based heading
                self.encoder_heading += math.degrees(angular_disp)

                # Update position
                if self.last_time:
                    dt = now - self.last_time

                    # Use fused heading for position update
                    heading_rad = math.radians(self.fused_heading)
                    self.pose.x += linear_dist * math.cos(heading_rad)
                    self.pose.y += linear_dist * math.sin(heading_rad)

                    # Calculate velocity
                    if dt > 0:
                        self.velocity.linear = linear_dist / dt
                        self.velocity.angular = angular_disp / dt

                self.last_left_count = enc_data.left_count
                self.last_right_count = enc_data.right_count

            # Update IMU heading
            if imu_data:
                self.imu_heading = imu_data.yaw

            # Fuse headings (complementary filter)
            # IMU is good for short-term changes, encoders for long-term
            self.fused_heading = (self.imu_weight * self.imu_heading +
                                  (1 - self.imu_weight) * self.encoder_heading)
            self.pose.theta = math.radians(self.fused_heading)
            self.pose.timestamp = now

            # Record path
            if len(self.path) == 0 or self._distance_from_last() > 10:  # Every 10mm
                self.path.append((self.pose.x, self.pose.y))
                if len(self.path) > self.max_path_points:
                    self.path.pop(0)

            self.last_time = now

            return NavigationState(
                pose=Pose(self.pose.x, self.pose.y, self.pose.theta, now),
                velocity=Velocity(self.velocity.linear, self.velocity.angular),
                imu_yaw=self.imu_heading,
                encoder_left=enc_data.left_count if enc_data else 0,
                encoder_right=enc_data.right_count if enc_data else 0,
                fused_heading=self.fused_heading
            )

    def _distance_from_last(self) -> float:
        """Distance from last recorded path point."""
        if not self.path:
            return float('inf')
        last = self.path[-1]
        return math.sqrt((self.pose.x - last[0])**2 + (self.pose.y - last[1])**2)

    def get_pose(self) -> Pose:
        """Get current pose."""
        with self._lock:
            return Pose(self.pose.x, self.pose.y, self.pose.theta, self.pose.timestamp)

    def get_heading(self) -> float:
        """Get current heading in degrees."""
        with self._lock:
            return self.fused_heading

    def get_velocity(self) -> Velocity:
        """Get current velocity."""
        with self._lock:
            return Velocity(self.velocity.linear, self.velocity.angular)

    def get_distance_traveled(self) -> float:
        """Get total distance traveled along path."""
        if len(self.path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.path)):
            dx = self.path[i][0] - self.path[i-1][0]
            dy = self.path[i][1] - self.path[i-1][1]
            total += math.sqrt(dx*dx + dy*dy)
        return total

    def start_continuous_update(self, rate_hz: float = 50):
        """Start background thread for continuous updates."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, args=(rate_hz,), daemon=True)
        self._thread.start()
        print(f"[Nav] Started continuous updates at {rate_hz}Hz")

    def stop_continuous_update(self):
        """Stop background updates."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None

    def _update_loop(self, rate_hz: float):
        """Background update loop."""
        period = 1.0 / rate_hz
        while self._running:
            self.update()
            time.sleep(period)

    def close(self):
        """Shutdown navigation."""
        self.stop_continuous_update()
        if self.imu:
            self.imu.close()
        if self.encoders:
            self.encoders.close()
        print("[Nav] Shutdown complete")


# Singleton
_nav: Optional[Navigation] = None


def get_navigation() -> Navigation:
    """Get or create global navigation instance."""
    global _nav
    if _nav is None:
        _nav = Navigation()
    return _nav


if __name__ == "__main__":
    # Test navigation
    nav = Navigation()

    if not nav.init_sensors(calibrate_imu=True):
        print("Failed to initialize sensors")
        exit(1)

    nav.reset()

    print("\nNavigation running (Ctrl+C to stop)")
    print("Move the robot around to see position updates")
    print("-" * 60)

    try:
        while True:
            state = nav.update()

            print(f"\rX:{state.pose.x:7.1f}mm  Y:{state.pose.y:7.1f}mm  "
                  f"Heading:{state.fused_heading:6.1f}deg  "
                  f"Vel:{state.velocity.linear:5.1f}mm/s  "
                  f"L:{state.encoder_left:5d} R:{state.encoder_right:5d}    ",
                  end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nFinal position:")
        pose = nav.get_pose()
        print(f"  X: {pose.x:.1f} mm")
        print(f"  Y: {pose.y:.1f} mm")
        print(f"  Heading: {math.degrees(pose.theta):.1f} degrees")
        print(f"  Distance traveled: {nav.get_distance_traveled():.1f} mm")

    finally:
        nav.close()
