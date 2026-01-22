#!/usr/bin/env python3
"""
Odometry Node with IMU Fusion

Fuses wheel encoder data with IMU gyroscope for accurate odometry.
- Encoders: linear distance (x, y position)
- IMU gyroscope: angular velocity (heading/yaw)

Uses complementary filter to combine encoder-derived heading with IMU gyro.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
import math


def quaternion_from_yaw(yaw):
    """Convert yaw angle to quaternion."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
        
        # Declare parameters
        self.declare_parameter('wheel_radius', 0.0381)  # 3" diameter = 76.2mm, radius = 38.1mm
        self.declare_parameter('wheel_base', 0.2286)    # 9" wheelbase = 228.6mm
        self.declare_parameter('ticks_per_rev', 360)    # Encoder ticks per wheel revolution
        self.declare_parameter('publish_tf', True)      # Publish odom->base_link transform
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('use_imu', True)         # Use IMU for heading
        self.declare_parameter('imu_alpha', 0.98)       # Complementary filter: trust IMU 98%
        
        # Get parameters
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.ticks_per_rev = self.get_parameter('ticks_per_rev').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.use_imu = self.get_parameter('use_imu').value
        self.imu_alpha = self.get_parameter('imu_alpha').value
        
        # Calculate meters per tick
        wheel_circumference = 2 * math.pi * self.wheel_radius
        self.meters_per_tick = wheel_circumference / self.ticks_per_rev
        
        self.get_logger().info(f'Wheel radius: {self.wheel_radius}m')
        self.get_logger().info(f'Wheel base: {self.wheel_base}m')
        self.get_logger().info(f'Ticks per rev: {self.ticks_per_rev}')
        self.get_logger().info(f'Meters per tick: {self.meters_per_tick:.6f}')
        self.get_logger().info(f'IMU fusion: {"enabled" if self.use_imu else "disabled"}')
        
        # State variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # Heading in radians
        self.last_left_ticks = None
        self.last_right_ticks = None
        self.last_time = None
        
        # IMU state
        self.imu_gyro_z = 0.0  # Latest gyro reading (rad/s)
        self.imu_received = False
        self.last_imu_time = None
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', 50)
        
        # TF broadcaster
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.encoder_sub = self.create_subscription(
            Int32MultiArray,
            'wheel_encoders',
            self.encoder_callback,
            10
        )
        
        if self.use_imu:
            self.imu_sub = self.create_subscription(
                Imu,
                'imu/data',
                self.imu_callback,
                10
            )
        
        # Timer to publish odom/TF at regular intervals (even when stationary)
        self.create_timer(0.05, self.publish_odom_tf)  # 20 Hz
        
        # Store last velocities for continuous publishing
        self.last_linear_vel = 0.0
        self.last_angular_vel = 0.0
        
        self.get_logger().info('Odometry node started')
    
    def imu_callback(self, msg: Imu):
        """Store latest IMU gyroscope reading for heading fusion."""
        self.imu_gyro_z = msg.angular_velocity.z
        self.imu_received = True
        self.last_imu_time = self.get_clock().now()
    
    def publish_odom_tf(self):
        """Publish odometry and TF at regular intervals."""
        current_time = self.get_clock().now()
        
        # Create and publish odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        
        # Set position
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = quaternion_from_yaw(self.theta)
        
        # Set velocity
        odom_msg.twist.twist.linear.x = self.last_linear_vel
        odom_msg.twist.twist.angular.z = self.last_angular_vel
        
        # Set covariance (lower = more confident)
        # Position covariance
        odom_msg.pose.covariance[0] = 0.01   # x
        odom_msg.pose.covariance[7] = 0.01   # y
        odom_msg.pose.covariance[14] = 1e6   # z (not measured)
        odom_msg.pose.covariance[21] = 1e6   # roll (not measured)
        odom_msg.pose.covariance[28] = 1e6   # pitch (not measured)
        odom_msg.pose.covariance[35] = 0.01 if self.imu_received else 0.05  # yaw
        
        # Velocity covariance
        odom_msg.twist.covariance[0] = 0.01   # linear x
        odom_msg.twist.covariance[7] = 0.01   # linear y
        odom_msg.twist.covariance[35] = 0.01 if self.imu_received else 0.05  # angular z
        
        self.odom_pub.publish(odom_msg)
        
        # Publish TF
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = current_time.to_msg()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation = quaternion_from_yaw(self.theta)
            self.tf_broadcaster.sendTransform(t)
    
    def encoder_callback(self, msg):
        if len(msg.data) < 2:
            return
        
        current_time = self.get_clock().now()
        left_ticks = msg.data[0]
        right_ticks = msg.data[1]
        
        # Initialize on first message
        if self.last_left_ticks is None:
            self.last_left_ticks = left_ticks
            self.last_right_ticks = right_ticks
            self.last_time = current_time
            return
        
        # Calculate tick deltas
        delta_left = left_ticks - self.last_left_ticks
        delta_right = right_ticks - self.last_right_ticks
        
        # Handle encoder overflow (32-bit signed)
        if abs(delta_left) > 1000000:
            delta_left = 0
        if abs(delta_right) > 1000000:
            delta_right = 0
        
        # Calculate time delta
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            dt = 0.02  # Fallback to 50Hz
        
        # Convert ticks to distance
        left_distance = delta_left * self.meters_per_tick
        right_distance = delta_right * self.meters_per_tick
        
        # Calculate linear distance (average of both wheels)
        linear_distance = (left_distance + right_distance) / 2.0
        
        # Calculate angular change
        # Option 1: From encoders (differential drive kinematics)
        encoder_angular = (right_distance - left_distance) / self.wheel_base
        
        # Option 2: From IMU gyroscope (more accurate for rotation)
        if self.use_imu and self.imu_received:
            imu_angular = self.imu_gyro_z * dt
            
            # Complementary filter: trust IMU for short-term, encoders for drift correction
            # alpha * imu + (1-alpha) * encoders
            angular_distance = self.imu_alpha * imu_angular + (1 - self.imu_alpha) * encoder_angular
            
            # Use IMU gyro directly for angular velocity (more accurate)
            self.last_angular_vel = self.imu_gyro_z
        else:
            angular_distance = encoder_angular
            self.last_angular_vel = encoder_angular / dt if dt > 0 else 0.0
        
        # Update pose using midpoint integration
        if abs(angular_distance) < 1e-6:
            # Straight line motion
            self.x += linear_distance * math.cos(self.theta)
            self.y += linear_distance * math.sin(self.theta)
        else:
            # Arc motion
            radius = linear_distance / angular_distance
            self.x += radius * (math.sin(self.theta + angular_distance) - math.sin(self.theta))
            self.y += radius * (math.cos(self.theta) - math.cos(self.theta + angular_distance))
        
        self.theta += angular_distance
        
        # Normalize theta to [-pi, pi]
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
        
        # Calculate linear velocity
        self.last_linear_vel = linear_distance / dt if dt > 0 else 0.0
        
        # Update state for next iteration
        self.last_left_ticks = left_ticks
        self.last_right_ticks = right_ticks
        self.last_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
