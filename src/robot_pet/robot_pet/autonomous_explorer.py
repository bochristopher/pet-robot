#!/usr/bin/env python3
"""
Autonomous Frontier Explorer for Nav2
Finds unexplored areas and sends navigation goals automatically.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String
import numpy as np
import math
import time
import json
from collections import deque


class AutonomousExplorer(Node):
    def __init__(self):
        super().__init__('autonomous_explorer')

        # Parameters
        self.declare_parameter('min_frontier_size', 10)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('exploration_timeout', 300.0)  # 5 minutes
        self.declare_parameter('benchmark_mode', True)

        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.exploration_timeout = self.get_parameter('exploration_timeout').value
        self.benchmark_mode = self.get_parameter('benchmark_mode').value

        # State
        self.map_data = None
        self.map_info = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.current_goal = None
        self.navigating = False
        self.exploration_complete = False

        # Benchmarking metrics
        self.start_time = time.time()
        self.metrics = {
            'start_time': self.start_time,
            'cells_mapped': [],  # (timestamp, count)
            'goals_sent': 0,
            'goals_reached': 0,
            'goals_failed': 0,
            'stuck_events': 0,
            'distance_traveled': 0.0,
            'last_position': None,
        }

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Publishers
        self.status_pub = self.create_publisher(String, '/explorer_status', 10)
        self.benchmark_pub = self.create_publisher(String, '/benchmark_data', 10)

        # Nav2 Action Client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Timer for exploration logic
        self.explore_timer = self.create_timer(2.0, self.explore_callback)

        # Timer for benchmark logging
        if self.benchmark_mode:
            self.benchmark_timer = self.create_timer(5.0, self.log_benchmark)

        self.get_logger().info('Autonomous Explorer started!')
        self.get_logger().info(f'Benchmark mode: {self.benchmark_mode}')

    def map_callback(self, msg: OccupancyGrid):
        """Store map data for frontier detection."""
        self.map_info = msg.info
        self.map_data = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width))

    def odom_callback(self, msg: Odometry):
        """Track robot position for distance calculation."""
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y

        # Calculate distance traveled
        if self.metrics['last_position'] is not None:
            dx = new_x - self.metrics['last_position'][0]
            dy = new_y - self.metrics['last_position'][1]
            self.metrics['distance_traveled'] += math.sqrt(dx*dx + dy*dy)

        self.metrics['last_position'] = (new_x, new_y)
        self.robot_x = new_x
        self.robot_y = new_y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_theta = math.atan2(siny_cosp, cosy_cosp)

    def find_frontiers(self):
        """Find frontier cells (unknown cells adjacent to free cells)."""
        if self.map_data is None:
            return []

        height, width = self.map_data.shape
        frontiers = []

        # Find frontier cells
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Check if unknown (-1)
                if self.map_data[y, x] != -1:
                    continue

                # Check if adjacent to free space (0)
                neighbors = [
                    self.map_data[y-1, x], self.map_data[y+1, x],
                    self.map_data[y, x-1], self.map_data[y, x+1]
                ]
                if any(n == 0 for n in neighbors):
                    frontiers.append((x, y))

        # Cluster frontiers
        clusters = self.cluster_frontiers(frontiers)

        # Filter by size and return centroids
        valid_clusters = []
        for cluster in clusters:
            if len(cluster) >= self.min_frontier_size:
                # Calculate centroid
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                valid_clusters.append((cx, cy, len(cluster)))

        return valid_clusters

    def cluster_frontiers(self, frontiers):
        """Group adjacent frontier cells into clusters."""
        if not frontiers:
            return []

        frontier_set = set(frontiers)
        clusters = []
        visited = set()

        for start in frontiers:
            if start in visited:
                continue

            # BFS to find connected frontier cells
            cluster = []
            queue = deque([start])

            while queue:
                cell = queue.popleft()
                if cell in visited:
                    continue
                if cell not in frontier_set:
                    continue

                visited.add(cell)
                cluster.append(cell)

                # Check 8-connected neighbors
                x, y = cell
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (x + dx, y + dy)
                        if neighbor not in visited and neighbor in frontier_set:
                            queue.append(neighbor)

            if cluster:
                clusters.append(cluster)

        return clusters

    def grid_to_world(self, gx, gy):
        """Convert grid coordinates to world coordinates."""
        if self.map_info is None:
            return 0.0, 0.0

        wx = self.map_info.origin.position.x + (gx + 0.5) * self.map_info.resolution
        wy = self.map_info.origin.position.y + (gy + 0.5) * self.map_info.resolution
        return wx, wy

    def select_best_frontier(self, frontiers):
        """Select the best frontier to explore (closest large one)."""
        if not frontiers:
            return None

        best = None
        best_score = float('inf')

        for fx, fy, size in frontiers:
            wx, wy = self.grid_to_world(fx, fy)

            # Distance to frontier
            dist = math.sqrt((wx - self.robot_x)**2 + (wy - self.robot_y)**2)

            # Score: prefer closer, larger frontiers
            # Lower is better
            score = dist / (math.log(size + 1) + 1)

            if score < best_score:
                best_score = score
                best = (wx, wy)

        return best

    def send_goal(self, x, y):
        """Send navigation goal to Nav2."""
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Nav2 action server not available')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f'Sending goal: ({x:.2f}, {y:.2f})')
        self.metrics['goals_sent'] += 1

        future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self.nav_feedback_callback)
        future.add_done_callback(self.goal_response_callback)

        self.current_goal = (x, y)
        self.navigating = True
        return True

    def goal_response_callback(self, future):
        """Handle goal acceptance/rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            self.navigating = False
            self.metrics['goals_failed'] += 1
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Handle navigation result."""
        result = future.result()
        self.navigating = False

        if result.status == 4:  # SUCCEEDED
            self.get_logger().info('Goal reached!')
            self.metrics['goals_reached'] += 1
        else:
            self.get_logger().warn(f'Goal failed with status: {result.status}')
            self.metrics['goals_failed'] += 1

    def nav_feedback_callback(self, feedback):
        """Handle navigation feedback."""
        pass  # Could log progress here

    def explore_callback(self):
        """Main exploration logic - find and go to frontiers."""
        # Check timeout
        elapsed = time.time() - self.start_time
        if elapsed > self.exploration_timeout:
            if not self.exploration_complete:
                self.get_logger().info('Exploration timeout reached')
                self.exploration_complete = True
                self.save_benchmark()
            return

        # Skip if still navigating
        if self.navigating:
            return

        # Skip if no map
        if self.map_data is None:
            return

        # Find frontiers
        frontiers = self.find_frontiers()

        if not frontiers:
            if not self.exploration_complete:
                self.get_logger().info('No more frontiers - exploration complete!')
                self.exploration_complete = True
                self.save_benchmark()
            return

        # Select and go to best frontier
        goal = self.select_best_frontier(frontiers)
        if goal:
            self.send_goal(goal[0], goal[1])

        # Publish status
        status = String()
        status.data = f'Frontiers: {len(frontiers)}, Goal: {goal}, Navigating: {self.navigating}'
        self.status_pub.publish(status)

    def count_mapped_cells(self):
        """Count number of known (non-unknown) cells."""
        if self.map_data is None:
            return 0
        return int(np.sum(self.map_data != -1))

    def log_benchmark(self):
        """Log benchmark metrics periodically."""
        elapsed = time.time() - self.start_time
        mapped = self.count_mapped_cells()

        self.metrics['cells_mapped'].append((elapsed, mapped))

        # Publish benchmark data
        data = {
            'elapsed_time': elapsed,
            'cells_mapped': mapped,
            'goals_sent': self.metrics['goals_sent'],
            'goals_reached': self.metrics['goals_reached'],
            'goals_failed': self.metrics['goals_failed'],
            'distance_traveled': self.metrics['distance_traveled'],
        }

        msg = String()
        msg.data = json.dumps(data)
        self.benchmark_pub.publish(msg)

        self.get_logger().info(
            f'[Benchmark] {elapsed:.1f}s | Mapped: {mapped} | '
            f'Goals: {self.metrics["goals_reached"]}/{self.metrics["goals_sent"]} | '
            f'Distance: {self.metrics["distance_traveled"]:.2f}m'
        )

    def save_benchmark(self):
        """Save final benchmark results."""
        elapsed = time.time() - self.start_time
        mapped = self.count_mapped_cells()

        results = {
            'system': 'ROS2_Nav2_Explorer',
            'total_time': elapsed,
            'final_cells_mapped': mapped,
            'goals_sent': self.metrics['goals_sent'],
            'goals_reached': self.metrics['goals_reached'],
            'goals_failed': self.metrics['goals_failed'],
            'distance_traveled': self.metrics['distance_traveled'],
            'mapping_history': self.metrics['cells_mapped'],
            'efficiency': mapped / max(self.metrics['distance_traveled'], 0.1),
        }

        # Save to file
        filename = f'/home/bo/ros2_ws/benchmark_ros2_{int(time.time())}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        self.get_logger().info(f'Benchmark saved to {filename}')
        self.get_logger().info(f'=== FINAL RESULTS ===')
        self.get_logger().info(f'Time: {elapsed:.1f}s')
        self.get_logger().info(f'Cells Mapped: {mapped}')
        self.get_logger().info(f'Distance: {self.metrics["distance_traveled"]:.2f}m')
        self.get_logger().info(f'Efficiency: {results["efficiency"]:.1f} cells/m')


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_benchmark()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
