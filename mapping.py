#!/usr/bin/env python3
"""
Real-time Mapping Module
Creates occupancy grid map from LiDAR + navigation data.
Displays map in terminal or graphical window.
"""

import time
import math
import sys
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from navigation import Navigation, Pose
from lidar_rplidar import LidarRPLidar, Scan


@dataclass
class MapConfig:
    """Map configuration."""
    width_mm: float = 10000      # Map width in mm (10m)
    height_mm: float = 10000     # Map height in mm (10m)
    resolution_mm: float = 50    # Cell size in mm (5cm)
    origin_x_mm: float = 5000    # Robot start X (center)
    origin_y_mm: float = 5000    # Robot start Y (center)
    max_range_mm: float = 6000   # Max LiDAR range to use


class OccupancyGrid:
    """2D occupancy grid map."""

    def __init__(self, config: MapConfig = MapConfig()):
        self.config = config

        # Calculate grid size
        self.width = int(config.width_mm / config.resolution_mm)
        self.height = int(config.height_mm / config.resolution_mm)

        # Occupancy grid: 0=unknown, >0=occupied count, <0=free count
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

        # Log-odds representation for probabilistic update
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.l_occ = 0.85   # Log-odds for occupied
        self.l_free = -0.4  # Log-odds for free
        self.l_max = 5.0
        self.l_min = -5.0

        print(f"[Map] Created {self.width}x{self.height} grid "
              f"({config.width_mm/1000:.1f}m x {config.height_mm/1000:.1f}m, "
              f"{config.resolution_mm}mm resolution)")

    def world_to_grid(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell."""
        gx = int((x_mm + self.config.origin_x_mm) / self.config.resolution_mm)
        gy = int((y_mm + self.config.origin_y_mm) / self.config.resolution_mm)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid cell to world coordinates."""
        x = gx * self.config.resolution_mm - self.config.origin_x_mm
        y = gy * self.config.resolution_mm - self.config.origin_y_mm
        return x, y

    def is_valid_cell(self, gx: int, gy: int) -> bool:
        """Check if cell is within grid bounds."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def update_from_scan(self, scan: Scan, pose: Pose):
        """Update map from LiDAR scan at given pose."""
        robot_x, robot_y = pose.x, pose.y
        robot_theta = pose.theta  # radians

        for point in scan.points:
            if point.distance <= 0 or point.distance > self.config.max_range_mm:
                continue

            # Calculate endpoint in world frame
            angle_world = robot_theta + math.radians(point.angle)
            end_x = robot_x + point.distance * math.cos(angle_world)
            end_y = robot_y + point.distance * math.sin(angle_world)

            # Get grid cells
            start_gx, start_gy = self.world_to_grid(robot_x, robot_y)
            end_gx, end_gy = self.world_to_grid(end_x, end_y)

            # Ray trace from robot to endpoint - mark as free
            for gx, gy in self._bresenham(start_gx, start_gy, end_gx, end_gy):
                if self.is_valid_cell(gx, gy):
                    self.log_odds[gy, gx] += self.l_free
                    self.log_odds[gy, gx] = max(self.l_min, self.log_odds[gy, gx])

            # Mark endpoint as occupied
            if self.is_valid_cell(end_gx, end_gy):
                self.log_odds[end_gy, end_gx] += self.l_occ
                self.log_odds[end_gy, end_gx] = min(self.l_max, self.log_odds[end_gy, end_gx])

    def _bresenham(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for ray tracing."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dy
                y0 += sy

        return points[:-1]  # Exclude endpoint (will be marked occupied)

    def get_occupancy(self, gx: int, gy: int) -> float:
        """Get occupancy probability (0-1) for cell."""
        if not self.is_valid_cell(gx, gy):
            return 0.5
        return 1.0 / (1.0 + math.exp(-self.log_odds[gy, gx]))

    def to_image(self) -> np.ndarray:
        """Convert to grayscale image (0=occupied, 127=unknown, 255=free)."""
        prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        img = ((1.0 - prob) * 255).astype(np.uint8)
        # Mark unknown as gray
        unknown = np.abs(self.log_odds) < 0.1
        img[unknown] = 127
        return img


class RealtimeMapper:
    """Real-time mapping with LiDAR and navigation."""

    def __init__(self, config: MapConfig = MapConfig()):
        self.config = config
        self.map = OccupancyGrid(config)
        self.nav: Optional[Navigation] = None
        self.lidar: Optional[LidarRPLidar] = None

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Stats
        self.scan_count = 0
        self.update_rate = 0.0

    def init_sensors(self) -> bool:
        """Initialize navigation and LiDAR."""
        try:
            print("[Mapper] Initializing sensors...")

            # Navigation (IMU + encoders)
            self.nav = Navigation()
            if not self.nav.init_sensors(calibrate_imu=True):
                print("[Mapper] Navigation init failed")
                return False

            # LiDAR
            self.lidar = LidarRPLidar()
            if not self.lidar.connected:
                print("[Mapper] LiDAR init failed")
                return False

            self.nav.reset()
            print("[Mapper] Sensors ready")
            return True

        except Exception as e:
            print(f"[Mapper] Init error: {e}")
            return False

    def start(self):
        """Start mapping."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._mapping_loop, daemon=True)
        self._thread.start()
        print("[Mapper] Mapping started")

    def stop(self):
        """Stop mapping."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[Mapper] Mapping stopped")

    def _mapping_loop(self):
        """Main mapping loop."""
        last_time = time.time()
        scan_times = []

        try:
            for scan in self.lidar.scan_generator():
                if not self._running:
                    break

                # Update navigation
                self.nav.update()
                pose = self.nav.get_pose()

                # Update map
                with self._lock:
                    self.map.update_from_scan(scan, pose)
                    self.scan_count += 1

                # Calculate update rate
                now = time.time()
                scan_times.append(now - last_time)
                if len(scan_times) > 10:
                    scan_times.pop(0)
                self.update_rate = 1.0 / (sum(scan_times) / len(scan_times)) if scan_times else 0
                last_time = now

        except Exception as e:
            print(f"[Mapper] Loop error: {e}")

    def get_map_image(self) -> np.ndarray:
        """Get current map as image."""
        with self._lock:
            return self.map.to_image()

    def display_terminal(self, width: int = 80, height: int = 30):
        """Display map in terminal using ASCII."""
        img = self.get_map_image()

        # Downsample to fit terminal
        h, w = img.shape
        step_x = max(1, w // width)
        step_y = max(1, h // height)

        pose = self.nav.get_pose() if self.nav else Pose()
        robot_gx, robot_gy = self.map.world_to_grid(pose.x, pose.y)

        # ASCII characters for different occupancy levels
        chars = " .-:=+*#%@"

        print("\033[H\033[J", end="")  # Clear screen
        print(f"=== REALTIME MAP === Scans: {self.scan_count} | Rate: {self.update_rate:.1f}Hz")
        print(f"Robot: X={pose.x:.0f}mm Y={pose.y:.0f}mm Heading={math.degrees(pose.theta):.1f}°")
        print("-" * width)

        for y in range(0, min(h, height * step_y), step_y):
            line = ""
            for x in range(0, min(w, width * step_x), step_x):
                # Check if robot position
                if abs(x - robot_gx) < step_x and abs(y - robot_gy) < step_y:
                    line += "R"
                else:
                    val = img[y, x]
                    if val == 127:  # Unknown
                        line += " "
                    else:
                        idx = int((255 - val) / 255 * (len(chars) - 1))
                        line += chars[idx]
            print(line)

    def save_map(self, filename: str = "map.png"):
        """Save map to image file."""
        try:
            from PIL import Image
            img = self.get_map_image()
            Image.fromarray(img).save(filename)
            print(f"[Mapper] Saved map to {filename}")
        except ImportError:
            # Save as numpy
            np.save(filename.replace('.png', '.npy'), self.map.log_odds)
            print(f"[Mapper] Saved map to {filename.replace('.png', '.npy')}")

    def close(self):
        """Shutdown mapper."""
        self.stop()
        if self.lidar:
            self.lidar.close()
        if self.nav:
            self.nav.close()
        print("[Mapper] Shutdown complete")


def main_terminal():
    """Run mapping with terminal display."""
    mapper = RealtimeMapper()

    if not mapper.init_sensors():
        print("Failed to initialize sensors")
        return

    mapper.start()
    time.sleep(0.5)  # Let it start

    print("\nMapping... Move the robot around!")
    print("Press Ctrl+C to stop and save map\n")
    time.sleep(1)

    try:
        while True:
            mapper.display_terminal(width=100, height=35)
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        mapper.save_map("robot_map.png")
        mapper.close()


def main_graphical():
    """Run mapping with graphical display."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    mapper = RealtimeMapper()

    if not mapper.init_sensors():
        print("Failed to initialize sensors")
        return

    mapper.start()
    time.sleep(0.5)

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 10))
    img_plot = ax.imshow(mapper.get_map_image(), cmap='gray', vmin=0, vmax=255)
    robot_dot, = ax.plot([], [], 'ro', markersize=10)
    ax.set_title("Real-time Map")

    def update(frame):
        img = mapper.get_map_image()
        img_plot.set_array(img)

        pose = mapper.nav.get_pose() if mapper.nav else Pose()
        gx, gy = mapper.map.world_to_grid(pose.x, pose.y)
        robot_dot.set_data([gx], [gy])

        ax.set_title(f"Scans: {mapper.scan_count} | {mapper.update_rate:.1f}Hz | "
                     f"Pos: ({pose.x:.0f}, {pose.y:.0f})mm")
        return img_plot, robot_dot

    ani = FuncAnimation(fig, update, interval=200, blit=True)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        mapper.save_map("robot_map.png")
        mapper.close()


if __name__ == "__main__":
    # Check for display
    import os
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        print("Starting graphical display...")
        main_graphical()
    else:
        print("No display detected, using terminal mode...")
        main_terminal()
