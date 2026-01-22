    #!/usr/bin/env python3
"""
LIDAR Filter Node - Cleans up LaserScan data

Applies temporal median filtering and spatial noise removal to reduce
spurious readings from RPLidar. Handles variable-length scans gracefully.

Optimized for indoor robot navigation:
- Clips distant readings that aren't useful
- Removes isolated noise points
- Temporal smoothing to reduce jitter
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from collections import deque


class LidarFilter(Node):
    def __init__(self):
        super().__init__("lidar_filter")
        
        self.declare_parameter("window_size", 3)
        self.declare_parameter("min_range", 0.15)
        self.declare_parameter("max_range", 4.0)  # 4m is plenty for indoor navigation
        self.declare_parameter("neighbor_threshold", 0.3)
        self.declare_parameter("min_neighbors", 2)
        self.declare_parameter("intensity_threshold", 0)  # Filter low-intensity returns
        
        self.window_size = self.get_parameter("window_size").value
        self.min_range = self.get_parameter("min_range").value
        self.max_range = self.get_parameter("max_range").value
        self.neighbor_threshold = self.get_parameter("neighbor_threshold").value
        self.min_neighbors = self.get_parameter("min_neighbors").value
        self.intensity_threshold = self.get_parameter("intensity_threshold").value
        
        # Store (length, ranges) tuples to track lengths
        self.scan_buffer = deque(maxlen=self.window_size)
        self.scan_count = 0

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        
        self.scan_pub = self.create_publisher(
            LaserScan, "/scan_filtered", 10
        )
        
        self.get_logger().info(
            f"LIDAR filter started (window={self.window_size}, "
            f"range={self.min_range}-{self.max_range}m, "
            f"noise_filter=neighbors>={self.min_neighbors})"
        )

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        current_length = len(ranges)
        
        # Get intensity data if available (for filtering weak returns)
        has_intensity = len(msg.intensities) == current_length
        if has_intensity:
            intensities = np.array(msg.intensities, dtype=np.float32)
        
        # Replace inf/nan with max_range (marks them as "no reading")
        ranges = np.where(np.isfinite(ranges), ranges, self.max_range)

        # Clip readings beyond useful range - distant readings are noise for indoor nav
        ranges = np.clip(ranges, self.min_range, self.max_range)
        
        # Filter out low-intensity returns (weak reflections = unreliable)
        if has_intensity and self.intensity_threshold > 0:
            weak_returns = intensities < self.intensity_threshold
            ranges[weak_returns] = self.max_range

        # Add to buffer with length tracking
        self.scan_buffer.append((current_length, ranges.copy()))

        # Temporal median filter - only use scans with matching length
        if len(self.scan_buffer) >= self.window_size:
            # Filter buffer to only include scans matching current length
            matching_scans = [
                scan for length, scan in self.scan_buffer 
                if length == current_length
            ]
            
            if len(matching_scans) >= 2:
                # Stack only matching-length arrays
                try:
                    stacked = np.stack(matching_scans, axis=0)
                    ranges = np.median(stacked, axis=0)
                except ValueError:
                    # Fallback: shapes still don't match (shouldn't happen)
                    pass
        
        # Remove isolated noise points (sporadic readings with no neighbors)
        filtered = self.remove_noise(ranges)
        
        # Publish filtered scan
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = self.min_range
        out.range_max = self.max_range
        out.ranges = filtered.tolist()
        # Don't forward intensities - they've been used for filtering
        out.intensities = []
        
        self.scan_pub.publish(out)
        
        # Log occasional stats
        self.scan_count += 1
        if self.scan_count % 500 == 0:
            valid_count = np.sum(filtered < self.max_range)
            self.get_logger().info(
                f"Processed {self.scan_count} scans | "
                f"{valid_count}/{current_length} valid points"
            )

    def remove_noise(self, ranges):
        """Remove isolated noise points using vectorized neighbor check."""
        n = len(ranges)
        if n == 0:
            return ranges
            
        filtered = ranges.copy()
        threshold = self.neighbor_threshold
        
        # Vectorized approach: check neighbors within window
        # For each point, count how many neighbors are within threshold
        neighbor_counts = np.zeros(n, dtype=np.int32)
        
        for offset in range(-3, 4):
            if offset == 0:
                continue
            # Shift and compare
            shifted = np.roll(ranges, offset)
            # Handle boundary - don't count wrapped values
            if offset > 0:
                shifted[:offset] = self.max_range + 1  # Make them not match
            else:
                shifted[offset:] = self.max_range + 1
            
            close_enough = np.abs(ranges - shifted) < threshold
            neighbor_counts += close_enough.astype(np.int32)
        
        # Mark points with too few neighbors as max_range
        # But only if they're not already at max_range
        valid_points = ranges < self.max_range
        isolated = (neighbor_counts < self.min_neighbors) & valid_points
        filtered[isolated] = self.max_range
        
        return filtered


def main(args=None):
    rclpy.init(args=args)
    node = LidarFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
