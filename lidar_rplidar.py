#!/usr/bin/env python3
"""
RPLIDAR A1/A2 Driver Module
Provides scan data from RPLIDAR sensor.
"""

import time
import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple, Generator
from rplidar import RPLidar

DEFAULT_PORT = "/dev/ttyUSB0"


@dataclass
class ScanPoint:
    """Single LiDAR scan point."""
    angle: float      # degrees (0-360)
    distance: float   # mm
    quality: int      # signal quality


@dataclass
class Scan:
    """Complete 360-degree scan."""
    points: List[ScanPoint]
    timestamp: float

    def to_cartesian(self) -> List[Tuple[float, float]]:
        """Convert to X,Y points in mm."""
        result = []
        for p in self.points:
            if p.distance > 0:
                rad = math.radians(p.angle)
                x = p.distance * math.cos(rad)
                y = p.distance * math.sin(rad)
                result.append((x, y))
        return result

    def get_distances(self, angle_min: float = 0, angle_max: float = 360) -> List[float]:
        """Get distances in angle range."""
        return [p.distance for p in self.points
                if angle_min <= p.angle <= angle_max and p.distance > 0]


class LidarRPLidar:
    """RPLIDAR driver with continuous scanning."""

    def __init__(self, port: str = DEFAULT_PORT):
        self.port = port
        self.lidar: Optional[RPLidar] = None
        self.connected = False

        # Latest scan
        self.current_scan: Optional[Scan] = None
        self._lock = threading.Lock()

        # Scanning thread
        self._running = False
        self._thread = None

        self._connect()

    def _connect(self):
        """Connect to RPLIDAR."""
        try:
            self.lidar = RPLidar(self.port)

            # Reset to clear any previous state
            self.lidar.stop()
            self.lidar.stop_motor()
            time.sleep(0.5)

            info = self.lidar.get_info()
            health = self.lidar.get_health()

            print(f"[LiDAR] Connected to RPLIDAR")
            print(f"        Model: {info['model']}, FW: {info['firmware']}")
            print(f"        Health: {health[0]}")

            if health[0] != 'Good':
                print(f"[LiDAR] Warning: Health status is {health[0]}")

            self.connected = True

        except Exception as e:
            print(f"[LiDAR] Connection failed: {e}")
            self.connected = False

    def get_scan(self) -> Optional[Scan]:
        """Get the latest complete scan."""
        with self._lock:
            return self.current_scan

    def scan_generator(self) -> Generator[Scan, None, None]:
        """Generator yielding complete scans."""
        if not self.connected:
            return

        scan_points = []
        last_angle = 0

        try:
            for measurement in self.lidar.iter_measures():
                new_scan, quality, angle, distance = measurement

                # New scan started
                if new_scan and len(scan_points) > 0:
                    scan = Scan(points=scan_points, timestamp=time.time())
                    with self._lock:
                        self.current_scan = scan
                    yield scan
                    scan_points = []

                # Add point to current scan
                if quality > 0 and distance > 0:
                    scan_points.append(ScanPoint(
                        angle=angle,
                        distance=distance,
                        quality=quality
                    ))

        except Exception as e:
            print(f"[LiDAR] Scan error: {e}")

    def start_scanning(self):
        """Start continuous scanning in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        print("[LiDAR] Scanning started")

    def stop_scanning(self):
        """Stop scanning."""
        self._running = False
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.stop_motor()
            except:
                pass
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        print("[LiDAR] Scanning stopped")

    def _scan_loop(self):
        """Background scanning loop."""
        try:
            for scan in self.scan_generator():
                if not self._running:
                    break
        except Exception as e:
            print(f"[LiDAR] Scan loop error: {e}")
        finally:
            self._running = False

    def get_front_distance(self, angle_range: float = 30) -> float:
        """Get minimum distance in front (0 degrees +/- range)."""
        scan = self.get_scan()
        if not scan:
            return float('inf')

        distances = []
        for p in scan.points:
            # Front is around 0 degrees (or 360)
            if p.distance > 0:
                if p.angle <= angle_range or p.angle >= (360 - angle_range):
                    distances.append(p.distance)

        return min(distances) if distances else float('inf')

    def get_sector_distances(self, sectors: int = 8) -> List[float]:
        """Get minimum distance in each sector around robot."""
        scan = self.get_scan()
        if not scan:
            return [float('inf')] * sectors

        sector_size = 360 / sectors
        min_distances = [float('inf')] * sectors

        for p in scan.points:
            if p.distance > 0:
                sector = int(p.angle / sector_size) % sectors
                if p.distance < min_distances[sector]:
                    min_distances[sector] = p.distance

        return min_distances

    def close(self):
        """Shutdown LiDAR."""
        self.stop_scanning()
        if self.lidar:
            try:
                self.lidar.disconnect()
            except:
                pass
            self.lidar = None
        self.connected = False
        print("[LiDAR] Disconnected")


# Singleton
_lidar: Optional[LidarRPLidar] = None


def get_lidar() -> LidarRPLidar:
    """Get or create global LiDAR instance."""
    global _lidar
    if _lidar is None:
        _lidar = LidarRPLidar()
    return _lidar


if __name__ == "__main__":
    # Test LiDAR
    lidar = LidarRPLidar()

    if not lidar.connected:
        print("Failed to connect to LiDAR")
        exit(1)

    print("\nScanning (Ctrl+C to stop)...")
    print("-" * 50)

    try:
        scan_count = 0
        for scan in lidar.scan_generator():
            scan_count += 1
            points = len(scan.points)
            front = lidar.get_front_distance()
            sectors = lidar.get_sector_distances(8)

            print(f"\rScan {scan_count}: {points} points, Front: {front:.0f}mm, "
                  f"Sectors: {[f'{d:.0f}' for d in sectors]}    ", end="", flush=True)

            if scan_count >= 100:
                break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        lidar.close()
