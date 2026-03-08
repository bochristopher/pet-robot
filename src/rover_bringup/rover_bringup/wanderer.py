#!/usr/bin/env python3
"""
Wanderer Node - Simple Obstacle Avoidance

Drives the robot around randomly while avoiding obstacles using:
- Ultrasonic sensors (front-left, front-right, back-center)
- LIDAR scan data
- Depth camera point cloud

This is useful for testing SLAM mapping autonomously.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range, LaserScan, PointCloud2
import random
import math
import struct


class Wanderer(Node):
    def __init__(self):
        super().__init__('wanderer')
        
        # Parameters
        self.declare_parameter('linear_speed', 0.15)  # m/s
        self.declare_parameter('angular_speed', 0.5)  # rad/s
        self.declare_parameter('obstacle_distance', 0.4)  # meters
        self.declare_parameter('use_lidar', True)
        self.declare_parameter('lidar_obstacle_distance', 0.35)  # meters
        self.declare_parameter('use_depth', True)
        self.declare_parameter('depth_obstacle_distance', 0.5)  # meters
        
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.obstacle_distance = self.get_parameter('obstacle_distance').value
        self.use_lidar = self.get_parameter('use_lidar').value
        self.lidar_obstacle_distance = self.get_parameter('lidar_obstacle_distance').value
        self.use_depth = self.get_parameter('use_depth').value
        self.depth_obstacle_distance = self.get_parameter('depth_obstacle_distance').value
        
        # Ultrasonic readings (in meters)
        self.us_front_left = float('inf')
        self.us_front_right = float('inf')
        self.us_back = float('inf')
        
        # LIDAR obstacle detection
        self.lidar_obstacle_left = False
        self.lidar_obstacle_right = False
        self.lidar_obstacle_front = False
        
        # Depth camera obstacle detection
        self.depth_obstacle_left = False
        self.depth_obstacle_center = False
        self.depth_obstacle_right = False
        self.depth_min_distance = float('inf')
        
        # State machine
        self.state = 'forward'  # forward, turn_left, turn_right, backup
        self.state_start_time = self.get_clock().now()
        self.turn_duration = 0.0
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Ultrasonic subscribers
        self.create_subscription(Range, 'ultrasonic/front_left', self.us_fl_callback, 10)
        self.create_subscription(Range, 'ultrasonic/front_right', self.us_fr_callback, 10)
        self.create_subscription(Range, 'ultrasonic/back_center', self.us_bc_callback, 10)
        
        # LIDAR subscriber
        if self.use_lidar:
            self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)
        
        # Depth camera subscriber (use BEST_EFFORT QoS to match camera publisher)
        if self.use_depth:
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.create_subscription(PointCloud2, 'camera/points', self.depth_callback, sensor_qos)
        
        # Control loop at 10Hz
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Wanderer started!')
        self.get_logger().info(f'  Linear speed: {self.linear_speed} m/s')
        self.get_logger().info(f'  Obstacle distance: {self.obstacle_distance} m')
        self.get_logger().info(f'  LIDAR: {"enabled" if self.use_lidar else "disabled"}')
        self.get_logger().info(f'  Depth camera: {"enabled" if self.use_depth else "disabled"}')
    
    def us_fl_callback(self, msg: Range):
        if msg.range > msg.min_range and msg.range < msg.max_range:
            self.us_front_left = msg.range
        else:
            self.us_front_left = float('inf')
    
    def us_fr_callback(self, msg: Range):
        if msg.range > msg.min_range and msg.range < msg.max_range:
            self.us_front_right = msg.range
        else:
            self.us_front_right = float('inf')
    
    def us_bc_callback(self, msg: Range):
        if msg.range > msg.min_range and msg.range < msg.max_range:
            self.us_back = msg.range
        else:
            self.us_back = float('inf')
    
    def lidar_callback(self, msg: LaserScan):
        """Check LIDAR for obstacles in front, left, and right sectors."""
        num_readings = len(msg.ranges)
        if num_readings == 0:
            return
        
        # Define sectors (assuming 360 degree LIDAR, 0 is forward)
        # Front: -30 to +30 degrees
        # Left: +30 to +90 degrees  
        # Right: -90 to -30 degrees
        
        angles_per_reading = (msg.angle_max - msg.angle_min) / num_readings
        
        front_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')
        
        for i, r in enumerate(msg.ranges):
            if r < msg.range_min or r > msg.range_max or math.isnan(r) or math.isinf(r):
                continue
            
            angle = msg.angle_min + i * angles_per_reading
            angle_deg = math.degrees(angle)
            
            # Normalize to -180 to 180
            while angle_deg > 180:
                angle_deg -= 360
            while angle_deg < -180:
                angle_deg += 360
            
            if -30 <= angle_deg <= 30:
                front_min = min(front_min, r)
            elif 30 < angle_deg <= 90:
                left_min = min(left_min, r)
            elif -90 <= angle_deg < -30:
                right_min = min(right_min, r)
        
        self.lidar_obstacle_front = front_min < self.lidar_obstacle_distance
        self.lidar_obstacle_left = left_min < self.lidar_obstacle_distance
        self.lidar_obstacle_right = right_min < self.lidar_obstacle_distance
    
    def depth_callback(self, msg: PointCloud2):
        """Check depth camera point cloud for obstacles."""
        # Parse point cloud to find minimum distances in left/center/right sectors
        # Point cloud is in camera_depth_optical_frame: Z forward, X right, Y down
        
        left_min = float('inf')
        center_min = float('inf')
        right_min = float('inf')
        
        # Get point cloud data
        point_step = msg.point_step
        row_step = msg.row_step
        data = msg.data
        
        # Find x, y, z field offsets
        x_offset = y_offset = z_offset = None
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
        
        if x_offset is None or z_offset is None:
            return
        
        # Sample points (don't process all for performance)
        num_points = msg.width * msg.height
        sample_step = max(1, num_points // 500)  # Sample ~500 points
        
        for i in range(0, num_points, sample_step):
            offset = i * point_step
            try:
                x = struct.unpack_from('f', data, offset + x_offset)[0]
                z = struct.unpack_from('f', data, offset + z_offset)[0]
            except struct.error:
                continue
            
            # Skip invalid points
            if math.isnan(x) or math.isnan(z) or z <= 0:
                continue
            
            # Z is forward distance, X is left/right
            # Divide into sectors based on X position
            if z < 2.0:  # Only consider points within 2m
                if x < -0.15:  # Right sector (X negative = right in optical frame)
                    right_min = min(right_min, z)
                elif x > 0.15:  # Left sector
                    left_min = min(left_min, z)
                else:  # Center sector
                    center_min = min(center_min, z)
        
        self.depth_obstacle_left = left_min < self.depth_obstacle_distance
        self.depth_obstacle_center = center_min < self.depth_obstacle_distance
        self.depth_obstacle_right = right_min < self.depth_obstacle_distance
        self.depth_min_distance = min(left_min, center_min, right_min)
    
    def control_loop(self):
        """Main control loop - state machine for obstacle avoidance."""
        cmd = Twist()
        
        now = self.get_clock().now()
        state_duration = (now - self.state_start_time).nanoseconds / 1e9
        
        # Check for obstacles from all sensors
        # Ultrasonic
        us_left = self.us_front_left < self.obstacle_distance
        us_right = self.us_front_right < self.obstacle_distance
        
        # LIDAR
        lidar_left = self.lidar_obstacle_left
        lidar_right = self.lidar_obstacle_right
        lidar_front = self.lidar_obstacle_front
        
        # Depth camera
        depth_left = self.depth_obstacle_left
        depth_right = self.depth_obstacle_right
        depth_center = self.depth_obstacle_center
        
        # Combine all sensors
        obstacle_front_left = us_left or lidar_left or depth_left
        obstacle_front_right = us_right or lidar_right or depth_right
        obstacle_front = obstacle_front_left or obstacle_front_right or lidar_front or depth_center
        obstacle_back = self.us_back < self.obstacle_distance
        
        # State machine
        if self.state == 'forward':
            if obstacle_front:
                # Choose turn direction based on which side has more space
                left_clear = not obstacle_front_left
                right_clear = not obstacle_front_right
                
                if left_clear and not right_clear:
                    self.state = 'turn_left'
                elif right_clear and not left_clear:
                    self.state = 'turn_right'
                elif self.us_front_left > self.us_front_right:
                    self.state = 'turn_left'
                elif self.us_front_right > self.us_front_left:
                    self.state = 'turn_right'
                else:
                    # Random choice if equal
                    self.state = random.choice(['turn_left', 'turn_right'])
                
                self.turn_duration = random.uniform(0.5, 1.5)
                self.state_start_time = now
                
                # Log which sensor triggered
                sensors = []
                if us_left or us_right:
                    sensors.append('ultrasonic')
                if lidar_left or lidar_right or lidar_front:
                    sensors.append('lidar')
                if depth_left or depth_right or depth_center:
                    sensors.append(f'depth({self.depth_min_distance:.2f}m)')
                self.get_logger().info(f'Obstacle [{", ".join(sensors)}]! Turning {"left" if "left" in self.state else "right"}')
            else:
                # Drive forward
                cmd.linear.x = self.linear_speed
                # Add slight random wandering
                cmd.angular.z = random.uniform(-0.1, 0.1)
        
        elif self.state == 'turn_left':
            if state_duration > self.turn_duration:
                self.state = 'forward'
                self.state_start_time = now
            else:
                cmd.angular.z = self.angular_speed
        
        elif self.state == 'turn_right':
            if state_duration > self.turn_duration:
                self.state = 'forward'
                self.state_start_time = now
            else:
                cmd.angular.z = -self.angular_speed
        
        elif self.state == 'backup':
            if state_duration > 1.0 or obstacle_back:
                self.state = random.choice(['turn_left', 'turn_right'])
                self.turn_duration = random.uniform(0.8, 1.5)
                self.state_start_time = now
            else:
                cmd.linear.x = -self.linear_speed * 0.5
        
        # Publish command
        self.cmd_pub.publish(cmd)
    
    def stop(self):
        """Stop the robot."""
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.get_logger().info('Wanderer stopped')


def main(args=None):
    rclpy.init(args=args)
    node = Wanderer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

