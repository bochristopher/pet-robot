#!/usr/bin/env python3
"""
Loop Closure Detection - Phase 4 of SLAM System

Detects when robot returns to previously visited locations
and applies corrections to reduce accumulated drift.
"""

import math
import time
import threading
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ScanDescriptor:
    """Compact representation of a scan for matching."""
    pose_x: float
    pose_y: float
    pose_theta: float
    timestamp: float
    histogram: np.ndarray  # Range histogram for quick comparison
    scan_id: int


@dataclass
class LoopCandidate:
    """A potential loop closure."""
    current_id: int
    match_id: int
    similarity: float
    dx: float  # Correction to apply
    dy: float
    dtheta: float


def compute_scan_histogram(scan: List[Tuple], num_bins: int = 36) -> np.ndarray:
    """
    Compute range histogram from scan.

    Divides scan into angular sectors and computes mean range per sector.
    This is a compact descriptor for quick scan comparison.

    Args:
        scan: LiDAR scan [(quality, angle_deg, dist_mm), ...]
        num_bins: Number of angular bins

    Returns:
        Histogram of mean ranges per sector
    """
    bin_size = 360.0 / num_bins
    ranges = [[] for _ in range(num_bins)]

    for _, angle_deg, dist_mm in scan:
        if dist_mm <= 0 or dist_mm > 12000:
            continue

        bin_idx = int(angle_deg / bin_size) % num_bins
        ranges[bin_idx].append(dist_mm / 1000.0)  # meters

    # Compute mean range per bin (0 if empty)
    histogram = np.zeros(num_bins)
    for i, r in enumerate(ranges):
        if r:
            histogram[i] = np.mean(r)

    return histogram


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute similarity between two scan histograms.

    Uses normalized correlation coefficient.

    Returns:
        Similarity score 0-1 (1 = identical)
    """
    if len(h1) != len(h2):
        return 0.0

    # Handle empty histograms
    if np.sum(h1) == 0 or np.sum(h2) == 0:
        return 0.0

    # Normalized correlation
    h1_norm = h1 - np.mean(h1)
    h2_norm = h2 - np.mean(h2)

    std1 = np.std(h1)
    std2 = np.std(h2)

    if std1 < 0.01 or std2 < 0.01:
        return 0.0

    correlation = np.sum(h1_norm * h2_norm) / (len(h1) * std1 * std2)

    # Also try rotated versions (robot may be facing different direction)
    best_corr = correlation
    for shift in range(1, len(h1)):
        h2_shifted = np.roll(h2_norm, shift)
        corr = np.sum(h1_norm * h2_shifted) / (len(h1) * std1 * std2)
        if corr > best_corr:
            best_corr = corr

    return max(0, min(1, (best_corr + 1) / 2))  # Map -1..1 to 0..1


class LoopClosureDetector:
    """
    Detects loop closures by comparing current scan to history.

    Maintains a database of past scans and finds matches
    when robot revisits locations.
    """

    def __init__(self,
                 min_travel_distance: float = 2.0,
                 min_travel_time: float = 30.0,
                 similarity_threshold: float = 0.7,
                 distance_threshold: float = 1.0,
                 max_history: int = 500):
        """
        Initialize loop closure detector.

        Args:
            min_travel_distance: Minimum distance before considering loop (m)
            min_travel_time: Minimum time before considering loop (s)
            similarity_threshold: Min histogram similarity for match
            distance_threshold: Max distance to consider revisit (m)
            max_history: Maximum scans to keep in history
        """
        self.min_travel_distance = min_travel_distance
        self.min_travel_time = min_travel_time
        self.similarity_threshold = similarity_threshold
        self.distance_threshold = distance_threshold
        self.max_history = max_history

        # Scan history
        self._history: deque = deque(maxlen=max_history)
        self._scan_id = 0

        # Cumulative distance traveled
        self._total_distance = 0.0
        self._last_pose: Optional[Tuple[float, float]] = None

        # Detected loops
        self._loops_detected = 0

        # Thread safety
        self._lock = threading.Lock()

    def add_scan(self, scan: List[Tuple],
                 pose_x: float, pose_y: float, pose_theta: float) -> Optional[LoopCandidate]:
        """
        Add scan to history and check for loop closure.

        Args:
            scan: Current LiDAR scan
            pose_x, pose_y, pose_theta: Current robot pose

        Returns:
            LoopCandidate if loop detected, None otherwise
        """
        # Compute histogram descriptor
        histogram = compute_scan_histogram(scan)

        # Create descriptor
        descriptor = ScanDescriptor(
            pose_x=pose_x,
            pose_y=pose_y,
            pose_theta=pose_theta,
            timestamp=time.time(),
            histogram=histogram,
            scan_id=self._scan_id
        )
        self._scan_id += 1

        # Update travel distance
        if self._last_pose is not None:
            dx = pose_x - self._last_pose[0]
            dy = pose_y - self._last_pose[1]
            self._total_distance += math.sqrt(dx*dx + dy*dy)
        self._last_pose = (pose_x, pose_y)

        with self._lock:
            # Check for loop closure
            loop = self._check_loop_closure(descriptor)

            # Add to history
            self._history.append(descriptor)

            return loop

    def _check_loop_closure(self, current: ScanDescriptor) -> Optional[LoopCandidate]:
        """Check if current scan matches any historical scan."""

        # Need minimum travel before checking
        if self._total_distance < self.min_travel_distance:
            return None

        best_match = None
        best_similarity = 0.0

        for past in self._history:
            # Skip recent scans (need time gap for true loop)
            time_gap = current.timestamp - past.timestamp
            if time_gap < self.min_travel_time:
                continue

            # Quick distance check
            dx = current.pose_x - past.pose_x
            dy = current.pose_y - past.pose_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > self.distance_threshold:
                continue

            # Compare histograms
            similarity = histogram_similarity(current.histogram, past.histogram)

            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = past

        if best_match:
            self._loops_detected += 1

            # Compute correction (difference between poses)
            dx = best_match.pose_x - current.pose_x
            dy = best_match.pose_y - current.pose_y
            dtheta = best_match.pose_theta - current.pose_theta

            # Normalize angle
            while dtheta > math.pi:
                dtheta -= 2 * math.pi
            while dtheta < -math.pi:
                dtheta += 2 * math.pi

            return LoopCandidate(
                current_id=current.scan_id,
                match_id=best_match.scan_id,
                similarity=best_similarity,
                dx=dx,
                dy=dy,
                dtheta=dtheta
            )

        return None

    def reset(self):
        """Clear history and reset state."""
        with self._lock:
            self._history.clear()
            self._scan_id = 0
            self._total_distance = 0.0
            self._last_pose = None
            self._loops_detected = 0

    def get_stats(self) -> dict:
        """Get detector statistics."""
        with self._lock:
            return {
                'history_size': len(self._history),
                'total_distance': self._total_distance,
                'loops_detected': self._loops_detected,
            }


class PoseGraph:
    """
    Simple pose graph for loop closure optimization.

    Stores poses and constraints, applies corrections when
    loop closures are detected.
    """

    def __init__(self):
        self._poses: List[Tuple[float, float, float]] = []  # (x, y, theta)
        self._constraints: List[Tuple[int, int, float, float, float]] = []  # (i, j, dx, dy, dtheta)

    def add_pose(self, x: float, y: float, theta: float) -> int:
        """Add a pose, return its index."""
        self._poses.append((x, y, theta))
        return len(self._poses) - 1

    def add_constraint(self, i: int, j: int, dx: float, dy: float, dtheta: float):
        """Add constraint between poses i and j."""
        self._constraints.append((i, j, dx, dy, dtheta))

    def optimize(self, loop: LoopCandidate) -> Tuple[float, float, float]:
        """
        Apply loop closure correction.

        For simplicity, we distribute the error linearly over recent poses.

        Returns:
            (dx, dy, dtheta) correction to apply to current pose
        """
        # Simple linear distribution of error
        # In a full implementation, this would use graph optimization (g2o, Ceres)

        # For now, just return a fraction of the correction
        # (conservative to avoid sudden jumps)
        correction_factor = 0.5

        return (
            loop.dx * correction_factor,
            loop.dy * correction_factor,
            loop.dtheta * correction_factor
        )

    def get_corrected_pose(self, current_pose: Tuple[float, float, float],
                           correction: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply correction to pose."""
        x, y, theta = current_pose
        dx, dy, dtheta = correction

        new_theta = theta + dtheta
        while new_theta > math.pi:
            new_theta -= 2 * math.pi
        while new_theta < -math.pi:
            new_theta += 2 * math.pi

        return (x + dx, y + dy, new_theta)


def test_loop_closure():
    """Test loop closure detection."""
    print("=" * 60)
    print("LOOP CLOSURE TEST")
    print("=" * 60)

    # Create detector
    detector = LoopClosureDetector(
        min_travel_distance=0.5,  # Low for testing
        min_travel_time=1.0,      # Low for testing
        similarity_threshold=0.6,
        distance_threshold=0.5
    )

    # Generate synthetic room scan
    def generate_scan(robot_x, robot_y, robot_theta):
        scan = []
        for angle in range(0, 360, 2):
            world_angle = math.radians(angle) + robot_theta

            dx = math.cos(world_angle)
            dy = math.sin(world_angle)

            # 4x4m room
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
                dist = min(valid) * 1000
                scan.append((50, angle, min(dist, 8000)))

        return scan

    # Test 1: No loop (different locations)
    print("\n1. Adding scans at different locations...")
    positions = [
        (0, 0, 0),
        (0.5, 0, 0),
        (1.0, 0, 0),
        (1.0, 0.5, math.pi/2),
        (1.0, 1.0, math.pi/2),
    ]

    for x, y, theta in positions:
        scan = generate_scan(x, y, theta)
        result = detector.add_scan(scan, x, y, theta)
        print(f"   Pos ({x:.1f}, {y:.1f}): loop={result is not None}")
        time.sleep(0.3)

    # Test 2: Return to start (should detect loop)
    print("\n2. Returning to start position...")
    time.sleep(1.5)  # Wait for min_travel_time

    # Return path
    return_positions = [
        (0.5, 1.0, math.pi),
        (0, 1.0, math.pi),
        (0, 0.5, -math.pi/2),
        (0.1, 0.1, 0.1),  # Close to start
    ]

    for x, y, theta in return_positions:
        scan = generate_scan(x, y, theta)
        result = detector.add_scan(scan, x, y, theta)
        if result:
            print(f"   LOOP DETECTED at ({x:.1f}, {y:.1f})!")
            print(f"   Matched scan {result.match_id}, similarity: {result.similarity:.3f}")
            print(f"   Correction: dx={result.dx:.3f}, dy={result.dy:.3f}, dtheta={math.degrees(result.dtheta):.1f}deg")
        else:
            print(f"   Pos ({x:.1f}, {y:.1f}): no loop")
        time.sleep(0.3)

    # Stats
    print("\n3. Statistics:")
    stats = detector.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.2f}")
        else:
            print(f"   {k}: {v}")

    # Test histogram similarity
    print("\n4. Histogram similarity test...")
    scan1 = generate_scan(0, 0, 0)
    scan2 = generate_scan(0, 0, 0)  # Same location
    scan3 = generate_scan(1, 1, 0)  # Different location

    h1 = compute_scan_histogram(scan1)
    h2 = compute_scan_histogram(scan2)
    h3 = compute_scan_histogram(scan3)

    print(f"   Same location similarity: {histogram_similarity(h1, h2):.3f}")
    print(f"   Different location similarity: {histogram_similarity(h1, h3):.3f}")

    print("\n" + "=" * 60)
    print("Loop closure test complete!")


if __name__ == "__main__":
    test_loop_closure()
