#!/usr/bin/env python3
"""
Odometry Module - Track robot position from motor movements.

Uses dead reckoning to estimate robot's (x, y, heading) position
based on wheel movements. This is essential for building maps.

Robot specs:
- Wheelbase: 267mm
- Needs calibration for accurate mapping
"""

import math
import time
from dataclasses import dataclass
from typing import Tuple, List


# Robot physical parameters (calibrate these!)
WHEELBASE = 0.267  # 267mm between wheels
WHEEL_SPEED = 0.15  # Approximate speed in m/s (calibrate!)
TURN_RATE = 1.0  # Approximate rad/s during turns (calibrate!)


@dataclass
class Pose:
    """Robot pose (position and orientation)."""
    x: float = 0.0  # meters
    y: float = 0.0  # meters
    theta: float = 0.0  # radians (0 = facing +X, pi/2 = facing +Y)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)

    def heading_degrees(self) -> float:
        return math.degrees(self.theta) % 360


class Odometry:
    """
    Dead reckoning odometry for robot position tracking.

    Usage:
        odom = Odometry()
        odom.move_forward(0.5)  # Moved forward for 0.5 seconds
        odom.turn_left(0.8)     # Turned left for 0.8 seconds
        print(odom.pose)        # Current position
    """

    def __init__(self):
        self.pose = Pose()
        self.history: List[Pose] = [Pose()]  # Track path
        self.total_distance = 0.0

        print("[Odom] ✅ Odometry initialized at origin (0, 0)")

    def reset(self):
        """Reset to origin."""
        self.pose = Pose()
        self.history = [Pose()]
        self.total_distance = 0.0
        print("[Odom] 🔄 Reset to origin")

    def move_forward(self, duration: float):
        """Update position after moving forward."""
        distance = WHEEL_SPEED * duration

        # Update position based on heading
        self.pose.x += distance * math.cos(self.pose.theta)
        self.pose.y += distance * math.sin(self.pose.theta)
        self.total_distance += distance

        self._record_pose()

    def move_backward(self, duration: float):
        """Update position after moving backward."""
        distance = WHEEL_SPEED * duration

        # Update position (negative direction)
        self.pose.x -= distance * math.cos(self.pose.theta)
        self.pose.y -= distance * math.sin(self.pose.theta)
        self.total_distance += distance

        self._record_pose()

    def turn_left(self, duration: float):
        """Update heading after turning left."""
        angle = TURN_RATE * duration
        self.pose.theta += angle
        self._normalize_theta()
        self._record_pose()

    def turn_right(self, duration: float):
        """Update heading after turning right."""
        angle = TURN_RATE * duration
        self.pose.theta -= angle
        self._normalize_theta()
        self._record_pose()

    def rotate_180(self):
        """Update heading after 180 degree rotation."""
        self.pose.theta += math.pi
        self._normalize_theta()
        self._record_pose()

    def _normalize_theta(self):
        """Keep theta in range [-pi, pi]."""
        while self.pose.theta > math.pi:
            self.pose.theta -= 2 * math.pi
        while self.pose.theta < -math.pi:
            self.pose.theta += 2 * math.pi

    def _record_pose(self):
        """Record current pose to history."""
        self.history.append(Pose(self.pose.x, self.pose.y, self.pose.theta))

    def get_pose(self) -> Pose:
        """Get current pose."""
        return self.pose

    def get_position(self) -> Tuple[float, float]:
        """Get current (x, y) position."""
        return (self.pose.x, self.pose.y)

    def get_heading(self) -> float:
        """Get current heading in radians."""
        return self.pose.theta

    def get_heading_degrees(self) -> float:
        """Get current heading in degrees."""
        return self.pose.heading_degrees()

    def get_path(self) -> List[Tuple[float, float]]:
        """Get path history as list of (x, y) points."""
        return [(p.x, p.y) for p in self.history]

    def __str__(self) -> str:
        return (f"Pose(x={self.pose.x:.2f}m, y={self.pose.y:.2f}m, "
                f"heading={self.pose.heading_degrees():.1f}°)")


# Singleton
_odometry = None

def get_odometry() -> Odometry:
    """Get or create odometry instance."""
    global _odometry
    if _odometry is None:
        _odometry = Odometry()
    return _odometry


if __name__ == "__main__":
    # Test odometry
    print("\n" + "="*50)
    print("🧭 Odometry Test")
    print("="*50)

    odom = Odometry()

    # Simulate a square path
    print("\nSimulating square path (1m sides):")

    for i in range(4):
        print(f"\n  Side {i+1}:")
        odom.move_forward(1.0 / WHEEL_SPEED)  # Move ~1 meter
        print(f"    After forward: {odom}")
        odom.turn_left(math.pi/2 / TURN_RATE)  # Turn 90 degrees
        print(f"    After turn: {odom}")

    print(f"\n  Final position: {odom}")
    print(f"  Total distance: {odom.total_distance:.2f}m")
    print(f"  Should be back at origin!")
    print("="*50)
