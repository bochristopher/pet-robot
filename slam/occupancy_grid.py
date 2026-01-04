#!/usr/bin/env python3
"""
Occupancy Grid Mapping for SLAM

2D probabilistic occupancy grid using log-odds representation.
Efficiently tracks free, occupied, and unknown space from LiDAR scans.
"""

import numpy as np
import threading
import time
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
from pathlib import Path


@dataclass
class MapMetadata:
    """Metadata for saved maps."""
    resolution: float      # meters per cell
    width: int            # cells
    height: int           # cells
    origin_x: float       # world x of cell (0,0)
    origin_y: float       # world y of cell (0,0)
    created: float        # timestamp
    updated: float        # last update timestamp


class OccupancyGrid:
    """
    2D Probabilistic Occupancy Grid Map.

    Uses log-odds representation for efficient updates:
    - log_odds > 0: more likely occupied
    - log_odds < 0: more likely free
    - log_odds = 0: unknown (50% probability)

    Attributes:
        resolution: meters per grid cell (default 0.02 = 2cm)
        width, height: grid dimensions in cells
        origin: world coordinates of grid center
    """

    # Log-odds parameters
    L_FREE = -0.4        # Log-odds update for free cells (seen through)
    L_OCCUPIED = 0.85    # Log-odds update for occupied cells (endpoint)
    L_MIN = -5.0         # Minimum log-odds (very confident free)
    L_MAX = 5.0          # Maximum log-odds (very confident occupied)
    L_PRIOR = 0.0        # Prior log-odds (unknown)

    # Probability thresholds
    P_OCCUPIED = 0.65    # Above this = occupied
    P_FREE = 0.35        # Below this = free

    def __init__(self,
                 width_meters: float = 20.0,
                 height_meters: float = 20.0,
                 resolution: float = 0.02,
                 origin: Tuple[float, float] = None):
        """
        Initialize occupancy grid.

        Args:
            width_meters: Map width in meters (default 20m)
            height_meters: Map height in meters (default 20m)
            resolution: Meters per cell (default 0.02 = 2cm)
            origin: World coordinates of map center (default robot start)
        """
        self.resolution = resolution
        self.width = int(width_meters / resolution)
        self.height = int(height_meters / resolution)

        # Origin is center of map in world coordinates
        if origin is None:
            self.origin_x = 0.0
            self.origin_y = 0.0
        else:
            self.origin_x, self.origin_y = origin

        # Log-odds grid (initialized to unknown/prior)
        self.grid = np.full((self.height, self.width), self.L_PRIOR, dtype=np.float32)

        # Thread safety
        self._lock = threading.RLock()

        # Metadata
        self.created = time.time()
        self.updated = time.time()
        self.scan_count = 0

        print(f"[OccupancyGrid] Created {self.width}x{self.height} grid "
              f"({width_meters}m x {height_meters}m, {resolution*100:.0f}cm resolution)")

    # ==================== Coordinate Transforms ====================

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to grid coordinates (cells).

        Args:
            x, y: World coordinates in meters

        Returns:
            (gx, gy): Grid cell coordinates
        """
        # Offset from origin, convert to cells
        gx = int((x - self.origin_x) / self.resolution + self.width / 2)
        gy = int((y - self.origin_y) / self.resolution + self.height / 2)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """
        Convert grid coordinates (cells) to world coordinates (meters).

        Args:
            gx, gy: Grid cell coordinates

        Returns:
            (x, y): World coordinates in meters
        """
        x = (gx - self.width / 2) * self.resolution + self.origin_x
        y = (gy - self.height / 2) * self.resolution + self.origin_y
        return x, y

    def is_in_bounds(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are within map bounds."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    # ==================== Probability Functions ====================

    @staticmethod
    def log_odds_to_prob(log_odds: float) -> float:
        """Convert log-odds to probability."""
        return 1.0 - 1.0 / (1.0 + np.exp(log_odds))

    @staticmethod
    def prob_to_log_odds(prob: float) -> float:
        """Convert probability to log-odds."""
        prob = np.clip(prob, 0.001, 0.999)
        return np.log(prob / (1.0 - prob))

    def get_probability(self, gx: int, gy: int) -> float:
        """Get occupancy probability for a cell."""
        if not self.is_in_bounds(gx, gy):
            return 0.5  # Unknown
        with self._lock:
            return self.log_odds_to_prob(self.grid[gy, gx])

    def get_probability_world(self, x: float, y: float) -> float:
        """Get occupancy probability at world coordinates."""
        gx, gy = self.world_to_grid(x, y)
        return self.get_probability(gx, gy)

    # ==================== Cell State Queries ====================

    def is_occupied(self, gx: int, gy: int) -> bool:
        """Check if cell is occupied (high probability)."""
        return self.get_probability(gx, gy) > self.P_OCCUPIED

    def is_free(self, gx: int, gy: int) -> bool:
        """Check if cell is free (low probability)."""
        return self.get_probability(gx, gy) < self.P_FREE

    def is_unknown(self, gx: int, gy: int) -> bool:
        """Check if cell is unknown (middle probability)."""
        p = self.get_probability(gx, gy)
        return self.P_FREE <= p <= self.P_OCCUPIED

    def is_occupied_world(self, x: float, y: float) -> bool:
        """Check if world position is occupied."""
        gx, gy = self.world_to_grid(x, y)
        return self.is_occupied(gx, gy)

    # ==================== Ray Tracing ====================

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm for ray tracing.

        Returns list of (x, y) cells along line from (x0,y0) to (x1,y1).
        """
        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            cells.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return cells

    def update_ray(self, robot_x: float, robot_y: float,
                   endpoint_x: float, endpoint_y: float,
                   hit: bool = True):
        """
        Update grid along a single ray from robot to endpoint.

        Args:
            robot_x, robot_y: Robot position in world coords
            endpoint_x, endpoint_y: Ray endpoint in world coords
            hit: True if endpoint hit obstacle, False if max range
        """
        # Convert to grid coordinates
        gx0, gy0 = self.world_to_grid(robot_x, robot_y)
        gx1, gy1 = self.world_to_grid(endpoint_x, endpoint_y)

        # Get cells along ray
        cells = self._bresenham_line(gx0, gy0, gx1, gy1)

        with self._lock:
            # Mark all cells except last as FREE (ray passed through)
            for gx, gy in cells[:-1]:
                if self.is_in_bounds(gx, gy):
                    self.grid[gy, gx] = np.clip(
                        self.grid[gy, gx] + self.L_FREE,
                        self.L_MIN, self.L_MAX
                    )

            # Mark last cell as OCCUPIED (if hit) or FREE (if max range)
            if cells:
                gx, gy = cells[-1]
                if self.is_in_bounds(gx, gy):
                    if hit:
                        self.grid[gy, gx] = np.clip(
                            self.grid[gy, gx] + self.L_OCCUPIED,
                            self.L_MIN, self.L_MAX
                        )
                    else:
                        self.grid[gy, gx] = np.clip(
                            self.grid[gy, gx] + self.L_FREE,
                            self.L_MIN, self.L_MAX
                        )

    def update_from_scan(self, robot_x: float, robot_y: float,
                         robot_theta: float,
                         scan: List[Tuple[float, float, float]],
                         max_range: float = 8.0,
                         min_range: float = 0.1):
        """
        Update occupancy grid from a full LiDAR scan.

        Args:
            robot_x, robot_y: Robot position in world coordinates (meters)
            robot_theta: Robot heading in radians (0 = +x axis)
            scan: List of (quality, angle_deg, distance_mm) tuples
            max_range: Maximum valid range in meters
            min_range: Minimum valid range in meters
        """
        for quality, angle_deg, distance_mm in scan:
            # Skip invalid readings
            if quality < 10 or distance_mm <= 0:
                continue

            distance = distance_mm / 1000.0  # Convert to meters

            # Check range validity
            if distance < min_range:
                continue

            hit = distance < max_range
            if not hit:
                distance = max_range  # Use max range for no-hit

            # Calculate endpoint in world coordinates
            # LiDAR angle is relative to robot, 0° = front
            angle_rad = math.radians(angle_deg) + robot_theta
            endpoint_x = robot_x + distance * math.cos(angle_rad)
            endpoint_y = robot_y + distance * math.sin(angle_rad)

            # Update ray
            self.update_ray(robot_x, robot_y, endpoint_x, endpoint_y, hit)

        with self._lock:
            self.scan_count += 1
            self.updated = time.time()

    # ==================== Visualization ====================

    def to_image(self) -> np.ndarray:
        """
        Convert grid to grayscale image for visualization.

        Returns:
            numpy array (height, width) with values 0-255
            - 0 (black) = occupied
            - 128 (gray) = unknown
            - 255 (white) = free
        """
        with self._lock:
            # Convert log-odds to probability
            prob = self.log_odds_to_prob(self.grid)

            # Convert to grayscale (inverted: high prob = dark)
            image = ((1.0 - prob) * 255).astype(np.uint8)

            return image

    def to_color_image(self, robot_pos: Tuple[float, float] = None,
                       robot_theta: float = None) -> np.ndarray:
        """
        Convert grid to color image with robot position.

        Colors:
            - Black: occupied
            - White: free
            - Gray: unknown
            - Red: robot position
            - Blue: robot heading indicator
        """
        with self._lock:
            prob = self.log_odds_to_prob(self.grid)

            # Create RGB image
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Free = white, Occupied = black, Unknown = gray
            gray = ((1.0 - prob) * 255).astype(np.uint8)
            image[:, :, 0] = gray
            image[:, :, 1] = gray
            image[:, :, 2] = gray

            # Draw robot position
            if robot_pos is not None:
                rx, ry = self.world_to_grid(robot_pos[0], robot_pos[1])
                if self.is_in_bounds(rx, ry):
                    # Red dot for robot
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if self.is_in_bounds(rx+dx, ry+dy):
                                image[ry+dy, rx+dx] = [255, 0, 0]

                    # Blue line for heading
                    if robot_theta is not None:
                        hx = int(rx + 15 * math.cos(robot_theta))
                        hy = int(ry + 15 * math.sin(robot_theta))
                        for t in range(20):
                            px = int(rx + t * (hx - rx) / 20)
                            py = int(ry + t * (hy - ry) / 20)
                            if self.is_in_bounds(px, py):
                                image[py, px] = [0, 0, 255]

            return image

    # ==================== Persistence ====================

    def save(self, filepath: str):
        """
        Save map to file (.npz format).

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            np.savez_compressed(
                filepath,
                grid=self.grid,
                resolution=self.resolution,
                width=self.width,
                height=self.height,
                origin_x=self.origin_x,
                origin_y=self.origin_y,
                created=self.created,
                updated=self.updated,
                scan_count=self.scan_count
            )

        print(f"[OccupancyGrid] Saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'OccupancyGrid':
        """
        Load map from file.

        Args:
            filepath: Path to load from

        Returns:
            Loaded OccupancyGrid instance
        """
        data = np.load(filepath)

        # Create grid with same parameters
        grid = cls(
            width_meters=float(data['width']) * float(data['resolution']),
            height_meters=float(data['height']) * float(data['resolution']),
            resolution=float(data['resolution']),
            origin=(float(data['origin_x']), float(data['origin_y']))
        )

        # Restore grid data
        grid.grid = data['grid']
        grid.created = float(data['created'])
        grid.updated = float(data['updated'])
        grid.scan_count = int(data['scan_count'])

        print(f"[OccupancyGrid] Loaded from {filepath} "
              f"({grid.scan_count} scans, last updated {time.ctime(grid.updated)})")

        return grid

    def save_image(self, filepath: str, robot_pos: Tuple[float, float] = None,
                   robot_theta: float = None):
        """Save map as PNG image."""
        import cv2

        if robot_pos is not None:
            image = self.to_color_image(robot_pos, robot_theta)
        else:
            image = self.to_image()
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Flip vertically for standard image coordinates
        image = cv2.flip(image, 0)

        cv2.imwrite(str(filepath), image)
        print(f"[OccupancyGrid] Saved image to {filepath}")

    # ==================== Map Analysis ====================

    def get_stats(self) -> dict:
        """Get map statistics."""
        with self._lock:
            prob = self.log_odds_to_prob(self.grid)

            occupied = np.sum(prob > self.P_OCCUPIED)
            free = np.sum(prob < self.P_FREE)
            unknown = self.width * self.height - occupied - free

            return {
                'width_m': self.width * self.resolution,
                'height_m': self.height * self.resolution,
                'resolution_cm': self.resolution * 100,
                'total_cells': self.width * self.height,
                'occupied_cells': int(occupied),
                'free_cells': int(free),
                'unknown_cells': int(unknown),
                'explored_pct': (occupied + free) / (self.width * self.height) * 100,
                'scan_count': self.scan_count,
                'last_updated': self.updated,
            }


def test_occupancy_grid():
    """Test occupancy grid with simulated data."""
    print("=" * 60)
    print("OCCUPANCY GRID TEST")
    print("=" * 60)

    # Create small test grid
    grid = OccupancyGrid(
        width_meters=10.0,
        height_meters=10.0,
        resolution=0.05  # 5cm for faster testing
    )

    # Test 1: Coordinate transforms
    print("\n1. Coordinate transforms:")
    world_x, world_y = 1.0, 2.0
    gx, gy = grid.world_to_grid(world_x, world_y)
    back_x, back_y = grid.grid_to_world(gx, gy)
    print(f"   World ({world_x}, {world_y}) -> Grid ({gx}, {gy}) -> World ({back_x:.2f}, {back_y:.2f})")

    # Test 2: Ray tracing
    print("\n2. Ray tracing:")
    grid.update_ray(0, 0, 2.0, 0, hit=True)  # Ray to the right
    grid.update_ray(0, 0, 0, 2.0, hit=True)  # Ray up
    print(f"   Updated rays, scan count: {grid.scan_count}")

    # Test 3: Simulated LiDAR scan
    print("\n3. Simulated LiDAR scan:")
    # Create a fake room: walls at 3m in all directions
    fake_scan = []
    for angle in range(0, 360, 2):
        # Simulate walls
        if angle < 45 or angle > 315:
            dist = 3000  # 3m front
        elif 45 <= angle < 135:
            dist = 2000  # 2m right
        elif 135 <= angle < 225:
            dist = 3500  # 3.5m back
        else:
            dist = 2500  # 2.5m left
        fake_scan.append((50, angle, dist))

    grid.update_from_scan(0, 0, 0, fake_scan)
    print(f"   Processed {len(fake_scan)} rays")

    # Test 4: Stats
    print("\n4. Map statistics:")
    stats = grid.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    # Test 5: Save and load
    print("\n5. Persistence:")
    test_path = "/tmp/test_map.npz"
    grid.save(test_path)

    loaded = OccupancyGrid.load(test_path)
    print(f"   Loaded grid: {loaded.width}x{loaded.height}, {loaded.scan_count} scans")

    # Test 6: Image export
    print("\n6. Image export:")
    img_path = "/tmp/test_map.png"
    grid.save_image(img_path, robot_pos=(0, 0), robot_theta=0)

    print("\n" + "=" * 60)
    print("Occupancy grid tests complete!")
    print(f"Map image saved to: {img_path}")


if __name__ == "__main__":
    test_occupancy_grid()
