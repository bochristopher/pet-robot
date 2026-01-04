#!/usr/bin/env python3
"""
SLAM System - Phase 5: Full Integration

Unified SLAM interface combining:
- Odometry (wheel encoders + IMU)
- Mapping (occupancy grid)
- Scan Matching (ICP for drift correction)
- Loop Closure (revisit detection)

Provides a single interface for smart_explore.py
"""

import time
import math
import threading
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Import SLAM components
from occupancy_grid import OccupancyGrid
from odometry import Odometry, Pose
from scan_matcher import ScanMatcher, ScanMatchResult
from loop_closure import LoopClosureDetector, LoopCandidate, PoseGraph


@dataclass
class SLAMPose:
    """Robot pose from SLAM system."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    x_std: float = 0.01
    y_std: float = 0.01
    theta_std: float = 0.02
    timestamp: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)


class SLAM:
    """
    Complete SLAM system.

    Fuses wheel odometry with LiDAR scan matching for accurate
    position tracking. Builds occupancy grid map and detects
    loop closures to correct drift.
    """

    def __init__(self,
                 arduino=None,
                 imu=None,
                 map_path: str = "/home/bo/robot_pet/slam/maps/slam_map.npz",
                 map_size: int = 1000,
                 resolution: float = 0.02,
                 use_scan_matching: bool = True,
                 use_loop_closure: bool = True,
                 auto_save_interval: float = 60.0):
        """
        Initialize SLAM system.

        Args:
            arduino: Serial connection for encoders
            imu: IMU sensor for heading
            map_path: Path to save/load map
            map_size: Grid size in cells
            resolution: Meters per cell
            use_scan_matching: Enable ICP scan matching
            use_loop_closure: Enable loop closure detection
            auto_save_interval: Seconds between auto-saves (0 to disable)
        """
        self.map_path = map_path
        self.use_scan_matching = use_scan_matching
        self.use_loop_closure = use_loop_closure
        self.auto_save_interval = auto_save_interval

        # Current pose
        self._pose = SLAMPose(timestamp=time.time())
        self._pose_lock = threading.RLock()

        # Components
        self._odom: Optional[Odometry] = None
        self._grid: Optional[OccupancyGrid] = None
        self._scan_matcher: Optional[ScanMatcher] = None
        self._loop_detector: Optional[LoopClosureDetector] = None
        self._pose_graph: Optional[PoseGraph] = None

        # Initialize odometry if arduino available
        if arduino is not None:
            self._odom = Odometry(
                arduino=arduino,
                imu=imu,
                wheel_base=0.20,
                wheel_diameter=0.065,
                ticks_per_rev=360
            )

        # Initialize map
        map_size_meters = map_size * resolution  # Convert cells to meters
        if map_path and Path(map_path).exists():
            try:
                self._grid = OccupancyGrid.load(map_path)
                print(f"[SLAM] Loaded existing map from {map_path}")
            except:
                self._grid = OccupancyGrid(
                    width_meters=map_size_meters,
                    height_meters=map_size_meters,
                    resolution=resolution
                )
        else:
            self._grid = OccupancyGrid(
                width_meters=map_size_meters,
                height_meters=map_size_meters,
                resolution=resolution
            )

        # Initialize scan matcher
        if use_scan_matching:
            self._scan_matcher = ScanMatcher(
                max_iterations=30,
                convergence_threshold=0.002,
                max_correspondence_dist=0.3,
                min_points=30
            )

        # Initialize loop closure
        if use_loop_closure:
            self._loop_detector = LoopClosureDetector(
                min_travel_distance=3.0,
                min_travel_time=60.0,
                similarity_threshold=0.7,
                distance_threshold=1.0
            )
            self._pose_graph = PoseGraph()

        # Threading
        self._running = False
        self._last_save_time = time.time()
        self._last_scan: Optional[List] = None

        # Statistics
        self._scans_processed = 0
        self._scan_matches = 0
        self._loop_closures = 0
        self._total_distance = 0.0

        # Callbacks
        self._on_pose_update: Optional[Callable[[SLAMPose], None]] = None
        self._on_loop_closure: Optional[Callable[[LoopCandidate], None]] = None

    def set_callbacks(self,
                      on_pose_update: Callable[[SLAMPose], None] = None,
                      on_loop_closure: Callable[[LoopCandidate], None] = None):
        """Set event callbacks."""
        self._on_pose_update = on_pose_update
        self._on_loop_closure = on_loop_closure

    def start(self):
        """Start SLAM system."""
        if self._running:
            return

        self._running = True

        # Start odometry
        if self._odom:
            self._odom.start()

        print("[SLAM] Started")

    def stop(self):
        """Stop SLAM system and save map."""
        self._running = False

        # Stop odometry
        if self._odom:
            self._odom.stop()

        # Save map
        self.save_map()

        print("[SLAM] Stopped")

    def update(self, scan: List[Tuple]) -> SLAMPose:
        """
        Process a new LiDAR scan.

        This is the main SLAM update function. Call this with each new scan.

        Args:
            scan: LiDAR scan [(quality, angle_deg, dist_mm), ...]

        Returns:
            Updated robot pose
        """
        self._scans_processed += 1

        # Get odometry estimate
        odom_pose = None
        if self._odom:
            odom_pose = self._odom.get_pose()

        with self._pose_lock:
            # Start with odometry pose or previous pose
            if odom_pose:
                new_x = odom_pose.x
                new_y = odom_pose.y
                new_theta = odom_pose.theta
            else:
                new_x = self._pose.x
                new_y = self._pose.y
                new_theta = self._pose.theta

            # Scan matching correction
            if self._scan_matcher and self._last_scan is not None:
                # Use odometry as initial guess
                if odom_pose and hasattr(self, '_last_odom_pose'):
                    dx_odom = odom_pose.x - self._last_odom_pose.x
                    dy_odom = odom_pose.y - self._last_odom_pose.y
                    dtheta_odom = odom_pose.theta - self._last_odom_pose.theta
                    initial_guess = (dx_odom, dy_odom, dtheta_odom)
                else:
                    initial_guess = (0, 0, 0)

                match_result = self._scan_matcher.match(scan, initial_guess)

                if match_result.converged and match_result.score > 0.5:
                    # Fuse scan matching with odometry
                    # Weight by confidence
                    alpha = 0.7  # Favor scan matching

                    if self._last_scan is not None:
                        # Apply scan match correction
                        new_x = self._pose.x + match_result.dx * alpha + (new_x - self._pose.x) * (1 - alpha)
                        new_y = self._pose.y + match_result.dy * alpha + (new_y - self._pose.y) * (1 - alpha)
                        new_theta = self._pose.theta + match_result.dtheta * alpha + (new_theta - self._pose.theta) * (1 - alpha)

                    self._scan_matches += 1

            # Store for next iteration
            self._last_scan = scan
            if odom_pose:
                self._last_odom_pose = odom_pose

            # Track distance
            dx = new_x - self._pose.x
            dy = new_y - self._pose.y
            self._total_distance += math.sqrt(dx*dx + dy*dy)

            # Loop closure check
            if self._loop_detector and self._pose_graph:
                loop = self._loop_detector.add_scan(scan, new_x, new_y, new_theta)

                if loop:
                    self._loop_closures += 1
                    print(f"[SLAM] Loop closure detected! similarity={loop.similarity:.2f}")

                    # Apply correction
                    correction = self._pose_graph.optimize(loop)
                    new_x += correction[0]
                    new_y += correction[1]
                    new_theta += correction[2]

                    if self._on_loop_closure:
                        self._on_loop_closure(loop)

            # Normalize theta
            while new_theta > math.pi:
                new_theta -= 2 * math.pi
            while new_theta < -math.pi:
                new_theta += 2 * math.pi

            # Update pose
            self._pose.x = new_x
            self._pose.y = new_y
            self._pose.theta = new_theta
            self._pose.timestamp = time.time()

            # Update map
            if self._grid:
                self._grid.update_from_scan(new_x, new_y, new_theta, scan)

        # Callback
        if self._on_pose_update:
            self._on_pose_update(self._pose)

        # Auto-save check
        if self.auto_save_interval > 0:
            if time.time() - self._last_save_time >= self.auto_save_interval:
                self.save_map()

        return self._pose

    def get_pose(self) -> SLAMPose:
        """Get current robot pose."""
        with self._pose_lock:
            return SLAMPose(
                x=self._pose.x,
                y=self._pose.y,
                theta=self._pose.theta,
                x_std=self._pose.x_std,
                y_std=self._pose.y_std,
                theta_std=self._pose.theta_std,
                timestamp=self._pose.timestamp
            )

    def set_pose(self, x: float, y: float, theta: float):
        """Manually set robot pose."""
        with self._pose_lock:
            self._pose.x = x
            self._pose.y = y
            self._pose.theta = theta
            self._pose.timestamp = time.time()

        if self._odom:
            self._odom.set_pose(x, y, theta)

        if self._scan_matcher:
            self._scan_matcher.reset()

        if self._loop_detector:
            self._loop_detector.reset()

    def reset(self):
        """Reset to origin."""
        self.set_pose(0, 0, 0)

        if self._odom:
            self._odom.reset()

        self._total_distance = 0.0
        self._scans_processed = 0
        self._scan_matches = 0
        self._loop_closures = 0

    def get_map(self) -> OccupancyGrid:
        """Get the occupancy grid map."""
        return self._grid

    def save_map(self):
        """Save map to disk."""
        if self._grid and self.map_path:
            try:
                Path(self.map_path).parent.mkdir(parents=True, exist_ok=True)
                self._grid.save(self.map_path)
                self._last_save_time = time.time()

                # Also save image
                img_path = str(self.map_path).replace('.npz', '.png')
                pose = self.get_pose()
                self._grid.save_image(img_path, robot_pos=(pose.x, pose.y), robot_theta=pose.theta)

                print(f"[SLAM] Map saved to {self.map_path}")
            except Exception as e:
                print(f"[SLAM] Save failed: {e}")

    def get_stats(self) -> dict:
        """Get SLAM statistics."""
        pose = self.get_pose()
        grid_stats = self._grid.get_stats() if self._grid else {}

        stats = {
            'pose_x': pose.x,
            'pose_y': pose.y,
            'pose_theta_deg': math.degrees(pose.theta),
            'total_distance_m': self._total_distance,
            'scans_processed': self._scans_processed,
            'scan_matches': self._scan_matches,
            'loop_closures': self._loop_closures,
            **grid_stats
        }

        if self._scan_matcher:
            stats.update({f'matcher_{k}': v for k, v in self._scan_matcher.get_stats().items()})

        if self._loop_detector:
            stats.update({f'loop_{k}': v for k, v in self._loop_detector.get_stats().items()})

        return stats

    # Convenience methods for exploration
    def is_explored(self, x: float, y: float) -> bool:
        """Check if a location has been explored (not unknown)."""
        if self._grid:
            gx, gy = self._grid.world_to_grid(x, y)
            if self._grid.is_in_bounds(gx, gy):
                return not self._grid.is_unknown(gx, gy)
        return False

    def is_obstacle(self, x: float, y: float) -> bool:
        """Check if a location is an obstacle."""
        if self._grid:
            return self._grid.is_occupied_world(x, y)
        return False

    def is_free(self, x: float, y: float) -> bool:
        """Check if a location is free space."""
        if self._grid:
            gx, gy = self._grid.world_to_grid(x, y)
            if self._grid.is_in_bounds(gx, gy):
                return self._grid.is_free(gx, gy)
        return False

    def get_frontiers(self) -> List[Tuple[float, float]]:
        """Get frontier cells (free cells adjacent to unknown)."""
        if not self._grid:
            return []

        frontiers = []
        # Scan grid for frontier cells
        for gx in range(1, self._grid.width - 1):
            for gy in range(1, self._grid.height - 1):
                if not self._grid.is_free(gx, gy):
                    continue

                # Check if adjacent to unknown
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self._grid.is_unknown(gx + dx, gy + dy):
                        wx, wy = self._grid.grid_to_world(gx, gy)
                        frontiers.append((wx, wy))
                        break

        return frontiers


def test_slam_simulated():
    """Test SLAM with simulated data."""
    print("=" * 60)
    print("SLAM SYSTEM TEST (SIMULATED)")
    print("=" * 60)

    # Generate synthetic room scan
    def generate_scan(robot_x, robot_y, robot_theta):
        scan = []
        for angle in range(0, 360, 2):
            world_angle = math.radians(angle) + robot_theta

            dx = math.cos(world_angle)
            dy = math.sin(world_angle)

            # 4x4m room with some obstacles
            distances = []

            # Walls
            if dx > 0.01:
                distances.append((2 - robot_x) / dx)
            if dx < -0.01:
                distances.append((-2 - robot_x) / dx)
            if dy > 0.01:
                distances.append((2 - robot_y) / dy)
            if dy < -0.01:
                distances.append((-2 - robot_y) / dy)

            valid = [d for d in distances if d > 0.1]
            if valid:
                dist = min(valid) * 1000
                scan.append((50, angle, min(dist, 8000)))

        return scan

    # Create SLAM (no hardware)
    slam = SLAM(
        arduino=None,
        imu=None,
        map_path="/tmp/test_slam_map.npz",
        use_scan_matching=True,
        use_loop_closure=False,  # Disable for simulated test
        auto_save_interval=0
    )
    slam.start()

    # Test 1: Stationary mapping
    print("\n1. Stationary mapping...")
    for i in range(5):
        scan = generate_scan(0, 0, 0)
        pose = slam.update(scan)
        print(f"   Scan {i+1}: pos=({pose.x:.3f}, {pose.y:.3f})")

    # Test 2: Moving robot
    print("\n2. Simulating movement...")
    path = [
        (0.2, 0, 0),
        (0.5, 0, 0),
        (0.5, 0.3, math.pi/4),
        (0.5, 0.5, math.pi/2),
        (0.3, 0.5, math.pi),
        (0, 0.5, math.pi),
    ]

    for x, y, theta in path:
        # Manually update pose (simulating odometry)
        slam.set_pose(x, y, theta)
        scan = generate_scan(x, y, theta)
        pose = slam.update(scan)
        print(f"   True: ({x:.2f}, {y:.2f}, {math.degrees(theta):.0f}deg) "
              f"| SLAM: ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.theta):.0f}deg)")

    # Test 3: Statistics
    print("\n3. Statistics:")
    stats = slam.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.3f}")
        else:
            print(f"   {k}: {v}")

    # Test 4: Map queries
    print("\n4. Map queries:")
    print(f"   Origin explored: {slam.is_explored(0, 0)}")
    print(f"   Far point explored: {slam.is_explored(5, 5)}")
    print(f"   Frontiers: {len(slam.get_frontiers())} cells")

    # Save
    slam.save_map()

    slam.stop()

    print("\n" + "=" * 60)
    print("SLAM test complete!")


if __name__ == "__main__":
    test_slam_simulated()
