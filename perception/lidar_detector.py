#!/usr/bin/env python3
"""
RPLidar A1M8 Obstacle Detection Module
Real-time 360° laser scanning for obstacle avoidance.

The RPLidar A1M8 provides:
- 360° scanning range
- 0.15m - 12m distance measurement
- ~5.5Hz scanning frequency
- 8000 samples/second
"""

import os
import sys
import time
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    from rplidar import RPLidar
except ImportError:
    print("❌ rplidar-roboticia not installed: pip3 install rplidar-roboticia")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# LiDAR device
LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUDRATE = 115200

# Robot physical dimensions (for reference)
ROBOT_WHEELBASE = 0.267  # 267mm between wheels
ROBOT_ANTENNA_PROTRUSION = 0.180  # 180mm antennas extend behind LiDAR center

# Obstacle detection zones (angles in degrees, 0° = front)
# Robot coordinate system: Front = 0°, Left = 90°, Right = 270° (or -90°)
# WIDENED ZONES for better obstacle detection!
ZONE_FRONT_MIN = 330  # -30° to +30° = 60° front zone
ZONE_FRONT_MAX = 30
ZONE_LEFT_MIN = 30
ZONE_LEFT_MAX = 150
ZONE_RIGHT_MIN = 210
ZONE_RIGHT_MAX = 330

# Dead zone for robot's own structure (antennas behind robot)
# Antennas extend 180mm behind LiDAR center - ignore rear readings < 250mm
REAR_DEAD_ZONE_MIN = 150  # 150° to 210° is the rear zone
REAR_DEAD_ZONE_MAX = 210
REAR_DEAD_ZONE_DISTANCE = 0.25  # Ignore readings < 250mm in rear (antenna + margin)

# Distance thresholds (meters)
OBSTACLE_CLOSE = 0.3  # 30cm - too close!
OBSTACLE_NEAR = 0.5   # 50cm - slow down
OBSTACLE_FAR = 1.0    # 1m - caution
CLEAR_DISTANCE = 1.5  # 1.5m - safe to move

# Minimum valid distance (RPLidar can report 0 or very small values for errors)
MIN_VALID_DISTANCE = 0.15  # 15cm minimum

# Quality threshold (RPLidar reports quality 0-15, higher = better)
MIN_QUALITY = 5  # Lowered from 10 to get more scan points


@dataclass
class ScanZone:
    """Represents obstacle info in a directional zone."""
    name: str  # "front", "left", "right"
    clear: bool  # Is path clear?
    min_distance: float  # Closest obstacle in meters
    avg_distance: float  # Average distance in zone
    num_points: int  # Number of scan points
    obstacle_level: str  # "clear", "far", "near", "close"
    suspicious: bool = False  # True if reading may be unreliable (reflective surface?)


@dataclass
class LidarScan:
    """Complete 360° scan result."""
    front: ScanZone
    left: ScanZone
    right: ScanZone
    timestamp: float
    total_points: int


class LidarDetector:
    """
    RPLidar A1M8 obstacle detector for autonomous navigation.

    Usage:
        detector = LidarDetector()
        detector.start()

        while True:
            scan = detector.get_scan()
            if scan.front.clear:
                print("Path clear ahead!")
    """

    def __init__(self, port: str = LIDAR_PORT):
        self.port = port
        self.lidar: Optional[RPLidar] = None
        self.running = False
        self.last_scan: Optional[LidarScan] = None
        self._scan_generator = None  # Persistent scan generator

        print(f"[LiDAR] Initializing on {port}...")

    def connect(self) -> bool:
        """Connect to LiDAR."""
        try:
            self.lidar = RPLidar(self.port)

            # Get device info
            info = self.lidar.get_info()
            health = self.lidar.get_health()

            print(f"[LiDAR] ✅ Connected!")
            print(f"[LiDAR]    Model: {info.get('model', 'Unknown')}")
            print(f"[LiDAR]    Firmware: {info.get('firmware', 'Unknown')}")
            print(f"[LiDAR]    Hardware: {info.get('hardware', 'Unknown')}")
            print(f"[LiDAR]    Health: {health[0]}")

            return True

        except Exception as e:
            print(f"[LiDAR] ❌ Connection failed: {e}")
            return False

    def start(self) -> bool:
        """Start scanning."""
        if not self.lidar:
            if not self.connect():
                return False

        try:
            print("[LiDAR] 🔄 Starting motor...")
            # Start motor and wait for it to stabilize
            # RPLidar library handles motor control automatically
            self.running = True
            print("[LiDAR] ✅ Scanning started!")
            return True

        except Exception as e:
            print(f"[LiDAR] ❌ Start failed: {e}")
            return False

    def stop(self):
        """Stop scanning and disconnect."""
        self.running = False
        self._scan_generator = None  # Clear the generator

        if self.lidar:
            try:
                print("[LiDAR] 🛑 Stopping...")
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
                print("[LiDAR] ✅ Stopped")
            except Exception as e:
                print(f"[LiDAR] ⚠️  Stop error: {e}")

            self.lidar = None

    def restart(self) -> bool:
        """Restart scanning (useful after errors)."""
        print("[LiDAR] 🔄 Restarting...")
        self.stop()
        time.sleep(0.5)
        return self.start()

    def _classify_distance(self, distance: float) -> str:
        """Classify distance level."""
        if distance < OBSTACLE_CLOSE:
            return "close"
        elif distance < OBSTACLE_NEAR:
            return "near"
        elif distance < OBSTACLE_FAR:
            return "far"
        else:
            return "clear"

    def _analyze_zone(self, points: List[Tuple[float, float]], zone_name: str) -> ScanZone:
        """
        Analyze scan points in a zone.

        Args:
            points: List of (angle, distance) tuples in degrees and meters
            zone_name: "front", "left", or "right"

        Returns:
            ScanZone with analysis
        """
        if not points:
            # No data in zone - ASSUME OBSTACLE for safety!
            # Could be reflective surface (glass, mirror) that doesn't return signal
            return ScanZone(
                name=zone_name,
                clear=False,
                min_distance=0.5,  # Assume obstacle at WARNING distance
                avg_distance=0.5,
                num_points=0,
                obstacle_level="near",
                suspicious=True  # Definitely suspicious - no data at all!
            )

        distances = [d for _, d in points]
        min_dist = min(distances)
        avg_dist = np.mean(distances)

        # REFLECTIVE SURFACE HANDLING:
        # If very few points in zone, LiDAR might be seeing a reflective surface
        # (glass, mirror, shiny metal) - be more conservative
        min_expected_points = 10  # Expect at least this many points per zone
        is_suspicious = len(points) < min_expected_points

        if is_suspicious:
            # Few points = unreliable data, might be reflective surface
            # Use more conservative distance estimate
            # If we got some readings, use 75% of min as safety margin
            min_dist = min(min_dist * 0.75, 1.0)  # Cap at 1m for safety
            avg_dist = min_dist

        obstacle_level = self._classify_distance(min_dist)
        is_clear = min_dist >= CLEAR_DISTANCE

        return ScanZone(
            name=zone_name,
            clear=is_clear,
            min_distance=min_dist,
            avg_distance=avg_dist,
            num_points=len(points),
            obstacle_level=obstacle_level,
            suspicious=is_suspicious
        )

    def get_scan(self, max_scans: int = 1) -> Optional[LidarScan]:
        """
        Get a complete 360° scan and analyze zones.

        Args:
            max_scans: Number of full rotations to accumulate (1 recommended for speed)

        Returns:
            LidarScan with obstacle info for each zone
        """
        if not self.running or not self.lidar:
            return None

        front_points = []
        left_points = []
        right_points = []
        total_points = 0

        try:
            # Use persistent scan generator to avoid repeated stop/start cycles
            if self._scan_generator is None:
                self._scan_generator = self.lidar.iter_scans(max_buf_meas=8000)

            # Get next scan from generator
            scan_count = 0
            for scan in self._scan_generator:
                scan_count += 1

                for quality, angle, distance in scan:
                    # Filter low quality and invalid distances
                    if quality < MIN_QUALITY or distance < MIN_VALID_DISTANCE:
                        continue

                    # Convert distance to meters
                    distance_m = distance / 1000.0

                    # Filter out robot's own antennas in rear dead zone
                    if (REAR_DEAD_ZONE_MIN <= angle <= REAR_DEAD_ZONE_MAX and
                        distance_m < REAR_DEAD_ZONE_DISTANCE):
                        continue  # Skip - this is probably the antenna

                    total_points += 1

                    # Classify into zones
                    # Front: -30° to +30° (or 330° to 360° and 0° to 30°) = 60° coverage
                    if angle <= ZONE_FRONT_MAX or angle >= ZONE_FRONT_MIN:
                        front_points.append((angle, distance_m))

                    # Left: 30° to 150° = 120° coverage
                    elif ZONE_LEFT_MIN <= angle <= ZONE_LEFT_MAX:
                        left_points.append((angle, distance_m))

                    # Right: 210° to 330° = 120° coverage
                    elif ZONE_RIGHT_MIN <= angle <= ZONE_RIGHT_MAX:
                        right_points.append((angle, distance_m))

                # Got enough scans?
                if scan_count >= max_scans:
                    break

            # Analyze each zone
            front_zone = self._analyze_zone(front_points, "front")
            left_zone = self._analyze_zone(left_points, "left")
            right_zone = self._analyze_zone(right_points, "right")

            scan_result = LidarScan(
                front=front_zone,
                left=left_zone,
                right=right_zone,
                timestamp=time.time(),
                total_points=total_points
            )

            self.last_scan = scan_result
            return scan_result

        except Exception as e:
            # Don't fail completely on scan errors - handle gracefully
            error_str = str(e).lower()

            # Known recoverable errors - handle silently
            recoverable_keywords = ["buffer", "descriptor", "unpack", "generator",
                                    "stop", "flags", "mismatch", "check bit"]

            if any(kw in error_str for kw in recoverable_keywords):
                # Reset generator and return cached scan
                self._scan_generator = None
                if self.last_scan and (time.time() - self.last_scan.timestamp) < 1.0:
                    return self.last_scan
            else:
                # Unexpected error - print it
                print(f"[LiDAR] ❌ Scan error: {e}")
                self._scan_generator = None  # Reset on any error

            # Return last scan if we have one (within 1.5 seconds)
            if self.last_scan and (time.time() - self.last_scan.timestamp) < 1.5:
                return self.last_scan

            return None

    def get_last_scan(self) -> Optional[LidarScan]:
        """Get the most recent scan result."""
        return self.last_scan

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Singleton instance
_detector: Optional[LidarDetector] = None


def get_detector() -> LidarDetector:
    """Get or create the global LiDAR detector instance."""
    global _detector
    if _detector is None:
        _detector = LidarDetector()
    return _detector


# =============================================================================
# CLI Testing
# =============================================================================

def main():
    """Test the LiDAR detector."""
    import argparse

    parser = argparse.ArgumentParser(description="RPLidar A1M8 Detector Test")
    parser.add_argument("--port", default=LIDAR_PORT, help="Serial port")
    parser.add_argument("--scans", type=int, default=5, help="Number of scans to display")
    parser.add_argument("--continuous", "-c", action="store_true", help="Continuous scanning")
    parser.add_argument("--fast", "-f", action="store_true", help="Fast mode (less delay between scans)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🔴 RPLidar A1M8 Obstacle Detector")
    print("="*60)

    detector = LidarDetector(port=args.port)

    if not detector.start():
        print("\n❌ Failed to start LiDAR")
        return

    try:
        scan_count = 0

        while args.continuous or scan_count < args.scans:
            scan = detector.get_scan()

            if scan:
                scan_count += 1

                print(f"\n📊 Scan #{scan_count} ({scan.total_points} points)")
                print("─" * 60)

                # Front
                status = "✅" if scan.front.clear else "🚧"
                print(f"   {status} FRONT:  {scan.front.min_distance:.2f}m "
                      f"(avg: {scan.front.avg_distance:.2f}m) "
                      f"[{scan.front.obstacle_level.upper()}] "
                      f"({scan.front.num_points} pts)")

                # Left
                status = "✅" if scan.left.clear else "🚧"
                print(f"   {status} LEFT:   {scan.left.min_distance:.2f}m "
                      f"(avg: {scan.left.avg_distance:.2f}m) "
                      f"[{scan.left.obstacle_level.upper()}] "
                      f"({scan.left.num_points} pts)")

                # Right
                status = "✅" if scan.right.clear else "🚧"
                print(f"   {status} RIGHT:  {scan.right.min_distance:.2f}m "
                      f"(avg: {scan.right.avg_distance:.2f}m) "
                      f"[{scan.right.obstacle_level.upper()}] "
                      f"({scan.right.num_points} pts)")

                # Navigation suggestion
                if scan.front.clear:
                    suggestion = "→ MOVE FORWARD"
                elif scan.left.clear and scan.left.min_distance > scan.right.min_distance:
                    suggestion = "↰ TURN LEFT"
                elif scan.right.clear:
                    suggestion = "↱ TURN RIGHT"
                else:
                    suggestion = "⟲ ROTATE 180°"

                print(f"\n   💡 Suggestion: {suggestion}")

            # Delay between scans
            if not args.fast:
                time.sleep(0.5)
            else:
                time.sleep(0.1)  # Faster for real-time operation

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    finally:
        detector.stop()
        print("\n✅ Test complete\n")


if __name__ == "__main__":
    main()
