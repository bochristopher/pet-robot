#!/usr/bin/env python3
"""
Occupancy Grid Map - Build 2D floor plan from LiDAR scans.

Each cell in the grid is marked as:
- UNKNOWN (0): Not yet explored
- FREE (1): Robot can pass through
- OCCUPIED (2): Wall or obstacle

The map is updated with each LiDAR scan based on robot's position.
"""

import math
import numpy as np
import json
import os
from datetime import datetime
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Map parameters
DEFAULT_MAP_SIZE = 20.0  # 20m x 20m default map
DEFAULT_RESOLUTION = 0.05  # 5cm per cell
MAP_ORIGIN_OFFSET = 10.0  # Start robot at center of map

# Cell values
UNKNOWN = 0
FREE = 1
OCCUPIED = 2


@dataclass
class MapConfig:
    """Map configuration."""
    size: float = DEFAULT_MAP_SIZE  # meters
    resolution: float = DEFAULT_RESOLUTION  # meters per cell
    origin_x: float = MAP_ORIGIN_OFFSET  # robot start X
    origin_y: float = MAP_ORIGIN_OFFSET  # robot start Y


class OccupancyMap:
    """
    2D Occupancy Grid Map for floor plan building.

    Usage:
        map = OccupancyMap()
        map.update(robot_x, robot_y, robot_heading, lidar_scan)
        map.save("floor_plan.json")
    """

    def __init__(self, config: Optional[MapConfig] = None):
        self.config = config or MapConfig()

        # Calculate grid dimensions
        self.width = int(self.config.size / self.config.resolution)
        self.height = int(self.config.size / self.config.resolution)

        # Initialize grid as unknown
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

        # Statistics
        self.updates = 0
        self.cells_explored = 0

        print(f"[Map] ✅ Occupancy map initialized")
        print(f"[Map]    Size: {self.config.size}m x {self.config.size}m")
        print(f"[Map]    Resolution: {self.config.resolution*100:.0f}cm/cell")
        print(f"[Map]    Grid: {self.width} x {self.height} cells")

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices."""
        # Offset so robot starts at center
        gx = int((x + self.config.origin_x) / self.config.resolution)
        gy = int((y + self.config.origin_y) / self.config.resolution)

        # Clamp to valid range
        gx = max(0, min(gx, self.width - 1))
        gy = max(0, min(gy, self.height - 1))

        return (gx, gy)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (meters)."""
        x = gx * self.config.resolution - self.config.origin_x
        y = gy * self.config.resolution - self.config.origin_y
        return (x, y)

    def update_from_lidar(self, robot_x: float, robot_y: float,
                          robot_theta: float, scan_points: List[Tuple[float, float]]):
        """
        Update map from a LiDAR scan.

        Args:
            robot_x, robot_y: Robot position in meters
            robot_theta: Robot heading in radians
            scan_points: List of (angle, distance) tuples from LiDAR
        """
        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)

        # Mark robot position as free
        self._set_cell(robot_gx, robot_gy, FREE)

        for angle_deg, distance in scan_points:
            if distance < 0.15 or distance > 8.0:  # Filter invalid readings
                continue

            # Convert to world coordinates
            angle_rad = math.radians(angle_deg) + robot_theta
            hit_x = robot_x + distance * math.cos(angle_rad)
            hit_y = robot_y + distance * math.sin(angle_rad)

            hit_gx, hit_gy = self.world_to_grid(hit_x, hit_y)

            # Ray trace from robot to hit point - mark cells as free
            self._ray_trace(robot_gx, robot_gy, hit_gx, hit_gy)

            # Mark hit point as occupied
            self._set_cell(hit_gx, hit_gy, OCCUPIED)

        self.updates += 1
        self._update_stats()

    def _ray_trace(self, x0: int, y0: int, x1: int, y1: int):
        """Mark cells along a ray as FREE using Bresenham's algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            # Don't overwrite the endpoint (it's occupied)
            if x == x1 and y == y1:
                break

            self._set_cell(x, y, FREE)

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _set_cell(self, gx: int, gy: int, value: int):
        """Set a cell value (with bounds checking)."""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            # Don't overwrite occupied with free
            if value == FREE and self.grid[gy, gx] == OCCUPIED:
                return
            self.grid[gy, gx] = value

    def _update_stats(self):
        """Update exploration statistics."""
        self.cells_explored = np.sum(self.grid != UNKNOWN)

    def get_explored_percentage(self) -> float:
        """Get percentage of map that has been explored."""
        total_cells = self.width * self.height
        return (self.cells_explored / total_cells) * 100

    def get_frontiers(self) -> List[Tuple[float, float]]:
        """
        Find frontier cells (free cells adjacent to unknown).
        These are the boundaries of explored area - where to explore next.
        """
        frontiers = []

        for gy in range(1, self.height - 1):
            for gx in range(1, self.width - 1):
                if self.grid[gy, gx] != FREE:
                    continue

                # Check if adjacent to unknown
                neighbors = [
                    self.grid[gy-1, gx], self.grid[gy+1, gx],
                    self.grid[gy, gx-1], self.grid[gy, gx+1]
                ]
                if UNKNOWN in neighbors:
                    world_x, world_y = self.grid_to_world(gx, gy)
                    frontiers.append((world_x, world_y))

        return frontiers

    def get_nearest_frontier(self, robot_x: float, robot_y: float) -> Optional[Tuple[float, float]]:
        """Get the nearest frontier point to explore."""
        frontiers = self.get_frontiers()
        if not frontiers:
            return None

        # Find nearest
        min_dist = float('inf')
        nearest = None

        for fx, fy in frontiers:
            dist = math.sqrt((fx - robot_x)**2 + (fy - robot_y)**2)
            if dist > 0.5 and dist < min_dist:  # At least 0.5m away
                min_dist = dist
                nearest = (fx, fy)

        return nearest

    def save(self, filepath: str = None) -> str:
        """Save map to file."""
        if filepath is None:
            os.makedirs("/home/bo/robot_pet/data/maps", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"/home/bo/robot_pet/data/maps/floor_plan_{timestamp}.json"

        data = {
            "config": {
                "size": self.config.size,
                "resolution": self.config.resolution,
                "origin_x": self.config.origin_x,
                "origin_y": self.config.origin_y,
            },
            "width": self.width,
            "height": self.height,
            "grid": self.grid.tolist(),
            "updates": self.updates,
            "cells_explored": int(self.cells_explored),
            "explored_percentage": self.get_explored_percentage(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

        print(f"[Map] 💾 Saved to {filepath}")
        return filepath

    def save_image(self, filepath: str = None) -> str:
        """Save map as PNG image."""
        try:
            import cv2
        except ImportError:
            print("[Map] ⚠️ OpenCV not available for image export")
            return None

        if filepath is None:
            os.makedirs("/home/bo/robot_pet/data/maps", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"/home/bo/robot_pet/data/maps/floor_plan_{timestamp}.png"

        # Create RGB image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Color mapping
        img[self.grid == UNKNOWN] = [128, 128, 128]  # Gray = unknown
        img[self.grid == FREE] = [255, 255, 255]     # White = free
        img[self.grid == OCCUPIED] = [0, 0, 0]       # Black = wall

        # Flip vertically for correct orientation
        img = cv2.flip(img, 0)

        cv2.imwrite(filepath, img)
        print(f"[Map] 🖼️ Image saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'OccupancyMap':
        """Load map from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        config = MapConfig(**data["config"])
        omap = cls(config)
        omap.grid = np.array(data["grid"], dtype=np.uint8)
        omap.updates = data["updates"]
        omap.cells_explored = data["cells_explored"]

        print(f"[Map] 📂 Loaded from {filepath}")
        return omap

    def __str__(self) -> str:
        return (f"OccupancyMap({self.width}x{self.height}, "
                f"{self.get_explored_percentage():.1f}% explored)")


if __name__ == "__main__":
    # Test
    print("\n" + "="*50)
    print("🗺️ Occupancy Map Test")
    print("="*50)

    omap = OccupancyMap()

    # Simulate some LiDAR readings at origin
    test_scan = [
        (0, 2.0),    # Front 2m
        (45, 1.5),   # Front-left 1.5m
        (90, 3.0),   # Left 3m
        (135, 2.5),  # Back-left 2.5m
        (180, 1.0),  # Back 1m
        (225, 2.0),  # Back-right 2m
        (270, 4.0),  # Right 4m
        (315, 1.8),  # Front-right 1.8m
    ]

    omap.update_from_lidar(0, 0, 0, test_scan)
    print(f"\nAfter first scan: {omap}")

    # Move and scan again
    omap.update_from_lidar(1.0, 0, 0, test_scan)
    print(f"After second scan: {omap}")

    frontiers = omap.get_frontiers()
    print(f"Frontiers found: {len(frontiers)}")

    # Save
    path = omap.save()
    img_path = omap.save_image()

    print("="*50)
