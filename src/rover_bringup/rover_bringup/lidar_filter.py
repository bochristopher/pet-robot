#!/usr/bin/env python3
"""
LIDAR Filter Node - Cleans up LaserScan data
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
        self.declare_parameter("max_range", 10.0)
        self.declare_parameter("neighbor_threshold", 0.3)
        self.declare_parameter("min_neighbors", 2)
        
        self.window_size = self.get_parameter("window_size").value
        self.min_range = self.get_parameter("min_range").value
        self.max_range = self.get_parameter("max_range").value
        self.neighbor_threshold = self.get_parameter("neighbor_threshold").value
        self.min_neighbors = self.get_parameter("min_neighbors").value
        
        self.scan_buffer = deque(maxlen=self.window_size)
        self.expected_length = None

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        
        self.scan_pub = self.create_publisher(
            LaserScan, "/scan_filtered", 10
        )
        
        self.get_logger().info("LIDAR filter started")

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Clear buffer if scan length changes
        if self.expected_length is not None and len(ranges) != self.expected_length:
            self.scan_buffer.clear()
        self.expected_length = len(ranges)

        # Replace inf/nan with max_range
        ranges = np.where(np.isfinite(ranges), ranges, self.max_range)

        # Clamp to valid range
        ranges = np.clip(ranges, self.min_range, self.max_range)

        # Add to buffer
        self.scan_buffer.append(ranges.copy())

        # Temporal median filter
        if len(self.scan_buffer) >= self.window_size:
            stacked = np.stack(self.scan_buffer, axis=0)
            ranges = np.median(stacked, axis=0)
        
        # Remove isolated noise points
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
        out.intensities = msg.intensities
        
        self.scan_pub.publish(out)

    def remove_noise(self, ranges):
        filtered = ranges.copy()
        n = len(ranges)
        
        for i in range(n):
            if ranges[i] >= self.max_range:
                continue
            
            neighbors = 0
            for j in range(max(0, i-3), min(n, i+4)):
                if i != j and abs(ranges[i] - ranges[j]) < self.neighbor_threshold:
                    neighbors += 1
            
            if neighbors < self.min_neighbors:
                filtered[i] = self.max_range
        
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
