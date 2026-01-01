#!/usr/bin/env python3
"""
Odometry Node - Publishes odometry from Arduino encoder data
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
import serial
import math
import time
import threading


def quaternion_from_yaw(yaw):
    """Create quaternion from yaw angle."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        # Parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('wheel_base', 0.17)
        self.declare_parameter('wheel_radius', 0.033)
        self.declare_parameter('ticks_per_rev', 1440)
        self.declare_parameter('publish_tf', True)

        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.ticks_per_rev = self.get_parameter('ticks_per_rev').value
        self.publish_tf = self.get_parameter('publish_tf').value

        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_left_ticks = None
        self.last_right_ticks = None
        self.last_time = time.time()

        # Connect to Arduino (can share with arduino_bridge if needed)
        try:
            self.arduino = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)
            self.get_logger().info(f'Connected to Arduino on {port}')
        except Exception as e:
            self.get_logger().warn(f'Could not open {port}: {e}')
            self.arduino = None

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)

        # TF broadcaster
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        # Timer to poll encoders
        self.timer = self.create_timer(0.05, self.update_odometry)  # 20Hz

        self.get_logger().info('Odometry Node ready!')

    def get_encoder_ticks(self):
        """Read encoder ticks from Arduino."""
        if self.arduino is None:
            return None, None

        try:
            self.arduino.write(b"ENCODERS\n")
            self.arduino.flush()
            line = self.arduino.readline().decode().strip()
            if line.startswith("ENC:"):
                parts = line[4:].split(",")
                left = int(parts[0])
                right = int(parts[1])
                return left, right
        except Exception as e:
            self.get_logger().debug(f'Encoder read error: {e}')
        return None, None

    def update_odometry(self):
        """Calculate and publish odometry."""
        left_ticks, right_ticks = self.get_encoder_ticks()

        if left_ticks is None or right_ticks is None:
            # If no encoder data, just publish current state
            self.publish_odom(0.0, 0.0)
            return

        # Initialize on first reading
        if self.last_left_ticks is None:
            self.last_left_ticks = left_ticks
            self.last_right_ticks = right_ticks
            return

        # Calculate tick deltas
        d_left = left_ticks - self.last_left_ticks
        d_right = right_ticks - self.last_right_ticks
        self.last_left_ticks = left_ticks
        self.last_right_ticks = right_ticks

        # Convert ticks to distance
        meters_per_tick = (2.0 * math.pi * self.wheel_radius) / self.ticks_per_rev
        left_dist = d_left * meters_per_tick
        right_dist = d_right * meters_per_tick

        # Calculate linear and angular displacement
        linear = (left_dist + right_dist) / 2.0
        angular = (right_dist - left_dist) / self.wheel_base

        # Update pose
        self.theta += angular
        self.x += linear * math.cos(self.theta)
        self.y += linear * math.sin(self.theta)

        # Calculate velocities
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        if dt > 0:
            linear_vel = linear / dt
            angular_vel = angular / dt
        else:
            linear_vel = 0.0
            angular_vel = 0.0

        self.publish_odom(linear_vel, angular_vel)

    def publish_odom(self, linear_vel, angular_vel):
        """Publish odometry message and TF."""
        now = self.get_clock().now()

        # Create odometry message
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # Position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = quaternion_from_yaw(self.theta)

        # Velocity
        odom.twist.twist.linear.x = linear_vel
        odom.twist.twist.angular.z = angular_vel

        self.odom_pub.publish(odom)

        # Publish TF
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = now.to_msg()
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_link'
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation = quaternion_from_yaw(self.theta)
            self.tf_broadcaster.sendTransform(t)


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
