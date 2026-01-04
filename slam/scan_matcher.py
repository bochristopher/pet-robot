#!/usr/bin/env python3
"""
Scan Matcher - Phase 3 of SLAM System

ICP (Iterative Closest Point) algorithm for matching LiDAR scans.
Computes transformation between consecutive scans to correct odometry drift.
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScanMatchResult:
    """Result of scan matching."""
    dx: float = 0.0         # Translation X (meters)
    dy: float = 0.0         # Translation Y (meters)
    dtheta: float = 0.0     # Rotation (radians)
    score: float = 0.0      # Match quality (0-1, higher is better)
    iterations: int = 0     # ICP iterations used
    converged: bool = False # Did ICP converge?


def scan_to_points(scan: List[Tuple], max_range: float = 8.0) -> np.ndarray:
    """
    Convert LiDAR scan to 2D point cloud.

    Args:
        scan: List of (quality, angle_deg, distance_mm) tuples
        max_range: Maximum range to include (meters)

    Returns:
        Nx2 numpy array of (x, y) points in robot frame
    """
    points = []
    for _, angle_deg, dist_mm in scan:
        if dist_mm <= 0 or dist_mm > max_range * 1000:
            continue

        dist = dist_mm / 1000.0
        angle = math.radians(angle_deg)

        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        points.append([x, y])

    if not points:
        return np.zeros((0, 2))

    return np.array(points)


def downsample_points(points: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    """
    Voxel grid downsampling to reduce point count.

    Args:
        points: Nx2 point cloud
        voxel_size: Grid cell size (meters)

    Returns:
        Downsampled point cloud
    """
    if len(points) == 0:
        return points

    # Quantize to grid
    grid_coords = np.floor(points / voxel_size).astype(int)

    # Get unique cells (keep one point per cell)
    _, unique_idx = np.unique(grid_coords, axis=0, return_index=True)

    return points[unique_idx]


def find_correspondences(source: np.ndarray, target: np.ndarray,
                         max_dist: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find nearest neighbor correspondences between point clouds.

    Args:
        source: Nx2 source points
        target: Mx2 target points
        max_dist: Maximum correspondence distance

    Returns:
        (source_matched, target_matched, distances)
    """
    if len(source) == 0 or len(target) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)

    # For each source point, find nearest target point
    # Using simple brute force (fast enough for typical scan sizes)
    src_matched = []
    tgt_matched = []
    dists = []

    for src_pt in source:
        # Compute distances to all target points
        diff = target - src_pt
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        # Find nearest
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist < max_dist:
            src_matched.append(src_pt)
            tgt_matched.append(target[min_idx])
            dists.append(min_dist)

    if not src_matched:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)

    return np.array(src_matched), np.array(tgt_matched), np.array(dists)


def compute_transform(source: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute optimal rigid transform (rotation + translation) between matched points.

    Uses SVD-based closed-form solution.

    Args:
        source: Nx2 source points
        target: Nx2 corresponding target points

    Returns:
        (dx, dy, dtheta) transformation from source to target
    """
    if len(source) < 3:
        return 0.0, 0.0, 0.0

    # Compute centroids
    src_centroid = np.mean(source, axis=0)
    tgt_centroid = np.mean(target, axis=0)

    # Center the points
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    # Compute cross-covariance matrix
    H = src_centered.T @ tgt_centered

    # SVD
    U, _, Vt = np.linalg.svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Extract rotation angle
    dtheta = math.atan2(R[1, 0], R[0, 0])

    # Translation
    t = tgt_centroid - R @ src_centroid
    dx, dy = t[0], t[1]

    return dx, dy, dtheta


def apply_transform(points: np.ndarray, dx: float, dy: float, dtheta: float) -> np.ndarray:
    """Apply rigid transform to points."""
    if len(points) == 0:
        return points

    cos_t = math.cos(dtheta)
    sin_t = math.sin(dtheta)

    # Rotate then translate
    x_new = points[:, 0] * cos_t - points[:, 1] * sin_t + dx
    y_new = points[:, 0] * sin_t + points[:, 1] * cos_t + dy

    return np.column_stack([x_new, y_new])


class ScanMatcher:
    """
    ICP-based scan matcher for LiDAR odometry.

    Matches consecutive scans to compute relative motion,
    correcting wheel odometry drift.
    """

    def __init__(self,
                 max_iterations: int = 50,
                 convergence_threshold: float = 0.001,
                 max_correspondence_dist: float = 0.5,
                 min_points: int = 20,
                 voxel_size: float = 0.03):
        """
        Initialize scan matcher.

        Args:
            max_iterations: Maximum ICP iterations
            convergence_threshold: Stop when transform change < this
            max_correspondence_dist: Max distance for point matching
            min_points: Minimum points required for matching
            voxel_size: Downsampling voxel size (0 to disable)
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_correspondence_dist = max_correspondence_dist
        self.min_points = min_points
        self.voxel_size = voxel_size

        # Previous scan for matching
        self._prev_scan: Optional[np.ndarray] = None

        # Statistics
        self._matches = 0
        self._failures = 0

    def match(self, scan: List[Tuple],
              initial_guess: Tuple[float, float, float] = (0, 0, 0)) -> ScanMatchResult:
        """
        Match current scan against previous scan.

        Args:
            scan: Current LiDAR scan
            initial_guess: (dx, dy, dtheta) initial transform estimate

        Returns:
            ScanMatchResult with computed transformation
        """
        # Convert to points
        current_points = scan_to_points(scan)

        if self.voxel_size > 0:
            current_points = downsample_points(current_points, self.voxel_size)

        # First scan - just store it
        if self._prev_scan is None:
            self._prev_scan = current_points
            return ScanMatchResult(converged=True, score=1.0)

        # Check minimum points
        if len(current_points) < self.min_points or len(self._prev_scan) < self.min_points:
            self._prev_scan = current_points
            self._failures += 1
            return ScanMatchResult()

        # Run ICP
        result = self._icp(current_points, self._prev_scan, initial_guess)

        # Update previous scan
        self._prev_scan = current_points

        if result.converged:
            self._matches += 1
        else:
            self._failures += 1

        return result

    def _icp(self, source: np.ndarray, target: np.ndarray,
             initial_guess: Tuple[float, float, float]) -> ScanMatchResult:
        """
        ICP algorithm implementation.

        Args:
            source: Current scan points
            target: Previous scan points
            initial_guess: Initial transform estimate

        Returns:
            ScanMatchResult
        """
        # Initialize transform
        dx, dy, dtheta = initial_guess
        total_dx, total_dy, total_dtheta = dx, dy, dtheta

        # Apply initial guess
        transformed = apply_transform(source, dx, dy, dtheta)

        prev_error = float('inf')

        for iteration in range(self.max_iterations):
            # Find correspondences
            src_matched, tgt_matched, dists = find_correspondences(
                transformed, target, self.max_correspondence_dist
            )

            if len(src_matched) < self.min_points:
                return ScanMatchResult(
                    dx=total_dx, dy=total_dy, dtheta=total_dtheta,
                    iterations=iteration + 1, converged=False
                )

            # Compute transform
            ddx, ddy, ddtheta = compute_transform(src_matched, tgt_matched)

            # Accumulate transform
            # Apply incremental rotation to accumulated translation
            cos_t = math.cos(ddtheta)
            sin_t = math.sin(ddtheta)
            new_dx = total_dx * cos_t - total_dy * sin_t + ddx
            new_dy = total_dx * sin_t + total_dy * cos_t + ddy
            total_dx, total_dy = new_dx, new_dy
            total_dtheta += ddtheta

            # Apply incremental transform
            transformed = apply_transform(transformed, ddx, ddy, ddtheta)

            # Check convergence
            change = math.sqrt(ddx**2 + ddy**2) + abs(ddtheta)
            mean_error = np.mean(dists)

            if change < self.convergence_threshold:
                # Compute match score (inverse of mean error, capped at 1)
                score = max(0, min(1, 1.0 - mean_error / self.max_correspondence_dist))

                return ScanMatchResult(
                    dx=total_dx, dy=total_dy, dtheta=total_dtheta,
                    score=score, iterations=iteration + 1, converged=True
                )

            prev_error = mean_error

        # Didn't converge but return best result
        score = max(0, min(1, 1.0 - prev_error / self.max_correspondence_dist))

        return ScanMatchResult(
            dx=total_dx, dy=total_dy, dtheta=total_dtheta,
            score=score, iterations=self.max_iterations, converged=False
        )

    def reset(self):
        """Reset matcher (clear previous scan)."""
        self._prev_scan = None

    def get_stats(self) -> dict:
        """Get matching statistics."""
        total = self._matches + self._failures
        return {
            'matches': self._matches,
            'failures': self._failures,
            'success_rate': self._matches / max(total, 1) * 100,
        }


def test_scan_matcher():
    """Test scan matcher with simulated data."""
    print("=" * 60)
    print("SCAN MATCHER TEST")
    print("=" * 60)

    # Create synthetic room scan (4x4m room)
    def generate_room_scan(robot_x, robot_y, robot_theta):
        scan = []
        for angle in range(0, 360, 2):
            world_angle = math.radians(angle) + robot_theta

            dx = math.cos(world_angle)
            dy = math.sin(world_angle)

            # Distance to walls
            distances = []
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
                dist = min(valid) * 1000  # mm
                scan.append((50, angle, min(dist, 8000)))

        return scan

    matcher = ScanMatcher(max_iterations=50, convergence_threshold=0.001)

    # Test 1: Match identical scans
    print("\n1. Identical scans (should give zero transform)...")
    scan1 = generate_room_scan(0, 0, 0)
    result1 = matcher.match(scan1)
    print(f"   First scan: {len(scan1)} points, converged={result1.converged}")

    scan2 = generate_room_scan(0, 0, 0)
    result2 = matcher.match(scan2)
    print(f"   Transform: dx={result2.dx:.4f}, dy={result2.dy:.4f}, dtheta={math.degrees(result2.dtheta):.2f}deg")
    print(f"   Score: {result2.score:.3f}, iterations: {result2.iterations}")

    # Test 2: Forward motion
    print("\n2. Forward motion (0.2m)...")
    matcher.reset()
    scan1 = generate_room_scan(0, 0, 0)
    matcher.match(scan1)

    scan2 = generate_room_scan(0.2, 0, 0)  # Robot moved forward 0.2m
    result = matcher.match(scan2)
    print(f"   Expected: dx=-0.2 (scan moves opposite to robot)")
    print(f"   Got:      dx={result.dx:.4f}, dy={result.dy:.4f}, dtheta={math.degrees(result.dtheta):.2f}deg")
    print(f"   Score: {result.score:.3f}, converged: {result.converged}")

    # Test 3: Rotation
    print("\n3. Rotation (10 degrees)...")
    matcher.reset()
    scan1 = generate_room_scan(0, 0, 0)
    matcher.match(scan1)

    scan2 = generate_room_scan(0, 0, math.radians(10))
    result = matcher.match(scan2)
    print(f"   Expected: dtheta=-10deg (scan rotates opposite)")
    print(f"   Got:      dtheta={math.degrees(result.dtheta):.2f}deg")
    print(f"   Score: {result.score:.3f}, converged: {result.converged}")

    # Test 4: Combined motion
    print("\n4. Combined motion (0.1m forward, 5deg left)...")
    matcher.reset()
    scan1 = generate_room_scan(0, 0, 0)
    matcher.match(scan1)

    scan2 = generate_room_scan(0.1, 0, math.radians(5))
    result = matcher.match(scan2)
    print(f"   Got: dx={result.dx:.4f}, dy={result.dy:.4f}, dtheta={math.degrees(result.dtheta):.2f}deg")
    print(f"   Score: {result.score:.3f}, converged: {result.converged}")

    # Stats
    print("\n5. Statistics:")
    stats = matcher.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("Scan matcher test complete!")


if __name__ == "__main__":
    test_scan_matcher()
