#!/usr/bin/env python3
"""
Real-time Mapper - Continuously builds occupancy grid from LiDAR

Runs in background thread, updates map from LiDAR scans,
auto-saves periodically, and provides visualization.
"""

import time
import math
import threading
from typing import Optional, Tuple, Callable
from pathlib import Path
import numpy as np

from occupancy_grid import OccupancyGrid


class Mapper:
    """
    Real-time occupancy grid mapper.

    Runs in background thread, consuming LiDAR scans and
    updating the occupancy grid with robot position.
    """

    def __init__(self,
                 grid: OccupancyGrid = None,
                 map_path: str = None,
                 auto_save_interval: float = 60.0):
        """
        Initialize mapper.

        Args:
            grid: Existing OccupancyGrid to use, or None to create new
            map_path: Path to save maps (auto-saves here)
            auto_save_interval: Seconds between auto-saves (0 to disable)
        """
        # Create or use existing grid
        if grid is not None:
            self.grid = grid
        elif map_path and Path(map_path).exists():
            self.grid = OccupancyGrid.load(map_path)
        else:
            self.grid = OccupancyGrid()

        self.map_path = map_path or "/home/bo/robot_pet/slam/maps/current_map.npz"
        self.auto_save_interval = auto_save_interval

        # Robot state (updated externally)
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_theta = 0.0
        self._pose_lock = threading.Lock()

        # LiDAR data queue
        self._scan_queue = []
        self._scan_lock = threading.Lock()

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._scans_processed = 0
        self._last_save = time.time()
        self._last_scan_time = 0

        # Callbacks
        self._on_map_update: Optional[Callable] = None

    def set_map_update_callback(self, callback: Callable):
        """Set callback called after each map update: callback(grid)"""
        self._on_map_update = callback

    def update_pose(self, x: float, y: float, theta: float):
        """
        Update robot pose (called from odometry).

        Args:
            x, y: Robot position in meters
            theta: Robot heading in radians
        """
        with self._pose_lock:
            self._robot_x = x
            self._robot_y = y
            self._robot_theta = theta

    def get_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose."""
        with self._pose_lock:
            return self._robot_x, self._robot_y, self._robot_theta

    def add_scan(self, scan: list):
        """
        Add LiDAR scan to processing queue.

        Args:
            scan: List of (quality, angle_deg, distance_mm) tuples
        """
        with self._scan_lock:
            # Keep only latest scan to avoid backlog
            self._scan_queue = [scan]
            self._last_scan_time = time.time()

    def start(self):
        """Start mapping thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._mapping_loop, daemon=True)
        self._thread.start()
        print("[Mapper] Started mapping thread")

    def stop(self):
        """Stop mapping thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Final save
        self.save()
        print("[Mapper] Stopped")

    def _mapping_loop(self):
        """Main mapping loop."""
        while self._running:
            # Get next scan
            scan = None
            with self._scan_lock:
                if self._scan_queue:
                    scan = self._scan_queue.pop(0)

            if scan is not None:
                # Get current pose
                x, y, theta = self.get_pose()

                # Update grid
                self.grid.update_from_scan(x, y, theta, scan)
                self._scans_processed += 1

                # Callback
                if self._on_map_update:
                    try:
                        self._on_map_update(self.grid)
                    except Exception as e:
                        pass

                # Auto-save check
                if self.auto_save_interval > 0:
                    if time.time() - self._last_save >= self.auto_save_interval:
                        self.save()

            else:
                # No scan available, sleep briefly
                time.sleep(0.01)

    def save(self):
        """Save current map to disk."""
        try:
            Path(self.map_path).parent.mkdir(parents=True, exist_ok=True)
            self.grid.save(self.map_path)
            self._last_save = time.time()

            # Also save image
            img_path = str(self.map_path).replace('.npz', '.png')
            x, y, theta = self.get_pose()
            self.grid.save_image(img_path, robot_pos=(x, y), robot_theta=theta)

        except Exception as e:
            print(f"[Mapper] Save failed: {e}")

    def get_stats(self) -> dict:
        """Get mapper statistics."""
        grid_stats = self.grid.get_stats()
        x, y, theta = self.get_pose()

        return {
            **grid_stats,
            'scans_processed': self._scans_processed,
            'robot_x': x,
            'robot_y': y,
            'robot_theta_deg': math.degrees(theta),
            'last_scan_age': time.time() - self._last_scan_time,
            'last_save_age': time.time() - self._last_save,
        }


class LiveMapVisualizer:
    """
    Real-time map visualization using OpenCV.
    """

    def __init__(self, mapper: Mapper, window_name: str = "SLAM Map"):
        self.mapper = mapper
        self.window_name = window_name
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start visualization thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._viz_loop, daemon=True)
        self._thread.start()
        print(f"[Visualizer] Started - window: {self.window_name}")

    def stop(self):
        """Stop visualization."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            import cv2
            cv2.destroyWindow(self.window_name)
        except:
            pass

    def _viz_loop(self):
        """Visualization loop."""
        import cv2

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 800)

        while self._running:
            try:
                # Get map image with robot position
                x, y, theta = self.mapper.get_pose()
                image = self.mapper.grid.to_color_image(
                    robot_pos=(x, y),
                    robot_theta=theta
                )

                # Flip for display
                image = cv2.flip(image, 0)

                # Add text overlay
                stats = self.mapper.get_stats()
                text = [
                    f"Pos: ({x:.2f}, {y:.2f}) {math.degrees(theta):.0f}deg",
                    f"Scans: {stats['scans_processed']}",
                    f"Explored: {stats['explored_pct']:.1f}%",
                ]
                y_offset = 30
                for line in text:
                    cv2.putText(image, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25

                cv2.imshow(self.window_name, image)

                key = cv2.waitKey(100)
                if key == ord('q'):
                    self._running = False
                elif key == ord('s'):
                    self.mapper.save()

            except Exception as e:
                time.sleep(0.1)

        cv2.destroyWindow(self.window_name)


def test_mapper_with_lidar():
    """Test mapper with real LiDAR data."""
    print("=" * 60)
    print("MAPPER TEST WITH LIDAR")
    print("=" * 60)

    import sys
    sys.path.insert(0, '/home/bo/robot_pet')

    # Initialize LiDAR
    print("\n1. Connecting to LiDAR...")
    try:
        from rplidar import RPLidar
        lidar = RPLidar('/dev/ttyUSB0')
        lidar.stop()
        lidar.stop_motor()
        time.sleep(0.5)
        lidar.clean_input()
        lidar.start_motor()
        time.sleep(2)
        print("   LiDAR connected")
    except Exception as e:
        print(f"   LiDAR failed: {e}")
        return

    # Create mapper
    print("\n2. Creating mapper...")
    mapper = Mapper(
        map_path="/home/bo/robot_pet/slam/maps/test_map.npz",
        auto_save_interval=30
    )
    mapper.start()

    # Start visualization
    print("\n3. Starting visualization...")
    viz = LiveMapVisualizer(mapper)
    viz.start()

    # Feed LiDAR data
    print("\n4. Mapping for 30 seconds...")
    print("   Move the robot around to build map")
    print("   Press 'q' in map window to quit early")
    print("   Press 's' to save map")

    try:
        start_time = time.time()
        scan_iter = lidar.iter_scans(max_buf_meas=8000, min_len=5)

        while viz._running and (time.time() - start_time) < 30:
            try:
                scan = next(scan_iter)
                mapper.add_scan(scan)

                # Simple simulated movement for testing
                t = time.time() - start_time
                x = 0.5 * math.sin(t * 0.2)  # Gentle oscillation
                y = 0.3 * math.cos(t * 0.15)
                theta = t * 0.1
                mapper.update_pose(x, y, theta)

            except StopIteration:
                lidar.stop()
                time.sleep(0.5)
                lidar.start_motor()
                time.sleep(1)
                scan_iter = lidar.iter_scans(max_buf_meas=8000, min_len=5)
            except Exception as e:
                print(f"   Scan error: {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n   Interrupted by user")

    # Cleanup
    print("\n5. Cleaning up...")
    viz.stop()
    mapper.stop()

    try:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
    except:
        pass

    # Final stats
    print("\n6. Final statistics:")
    stats = mapper.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("Mapper test complete!")
    print(f"Map saved to: {mapper.map_path}")


def test_mapper_simulated():
    """Test mapper with simulated data (no hardware needed)."""
    print("=" * 60)
    print("MAPPER TEST (SIMULATED)")
    print("=" * 60)

    # Create mapper
    mapper = Mapper(
        map_path="/tmp/simulated_map.npz",
        auto_save_interval=0  # Disable auto-save for test
    )
    mapper.start()

    # Simulate robot driving in a square
    print("\n1. Simulating square path...")

    # Simulated room: walls at different distances
    def generate_scan(robot_x, robot_y, robot_theta):
        scan = []
        for angle in range(0, 360, 2):
            world_angle = math.radians(angle) + robot_theta

            # Simple room: 4m x 4m centered at origin
            # Find intersection with walls
            dx = math.cos(world_angle)
            dy = math.sin(world_angle)

            # Distance to each wall
            distances = []
            if dx > 0.01:
                distances.append((2 - robot_x) / dx)  # Right wall
            if dx < -0.01:
                distances.append((-2 - robot_x) / dx)  # Left wall
            if dy > 0.01:
                distances.append((2 - robot_y) / dy)  # Top wall
            if dy < -0.01:
                distances.append((-2 - robot_y) / dy)  # Bottom wall

            # Take nearest positive distance
            valid = [d for d in distances if d > 0]
            if valid:
                dist = min(valid) * 1000  # Convert to mm
                dist = min(dist, 8000)  # Max range
                scan.append((50, angle, dist))

        return scan

    # Drive square path
    path = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, math.pi/2),
        (0, 1, math.pi),
        (0, 0, -math.pi/2),
    ]

    for x, y, theta in path:
        mapper.update_pose(x, y, theta)
        scan = generate_scan(x, y, theta)
        mapper.add_scan(scan)
        time.sleep(0.1)

    time.sleep(0.5)  # Let mapper process

    # Stats
    print("\n2. Results:")
    stats = mapper.get_stats()
    print(f"   Scans processed: {stats['scans_processed']}")
    print(f"   Explored: {stats['explored_pct']:.1f}%")
    print(f"   Occupied cells: {stats['occupied_cells']}")
    print(f"   Free cells: {stats['free_cells']}")

    # Save
    mapper.save()
    print(f"\n3. Map saved to: {mapper.map_path}")

    mapper.stop()
    print("\nSimulated test complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--simulated":
        test_mapper_simulated()
    else:
        test_mapper_with_lidar()
