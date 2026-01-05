#!/usr/bin/env python3
"""
Arduino Bridge Node - Bridges ROS2 cmd_vel to Arduino motor commands
WITH OBSTACLE AVOIDANCE from LiDAR AND Ultrasonic sensors
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Range
from std_msgs.msg import String, Float32MultiArray
import serial
import time
import threading
import math


class ArduinoBridge(Node):
    def __init__(self):
        super().__init__('arduino_bridge')

        # Parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('min_obstacle_dist', 0.25)  # meters

        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        self.min_obstacle_dist = self.get_parameter('min_obstacle_dist').value

        # Obstacle detection from sensors
        self.front_clear = True
        self.back_clear = True
        self.left_clear = True
        self.right_clear = True

        # Ultrasonic readings (in meters, -1 = no reading)
        self.us_front_left = -1.0
        self.us_front_right = -1.0
        self.us_back = -1.0

        # Connect to Arduino
        self.arduino = None
        try:
            self.arduino = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)  # Wait for Arduino reset
            self.get_logger().info(f'Connected to Arduino on {port}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to Arduino: {e}')

        # Camera obstacle scores (left, center, right)
        self.cam_obstacles = [0.0, 0.0, 0.0]

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.cam_sub = self.create_subscription(
            Float32MultiArray, 'camera/obstacles', self.camera_callback, 10)

        # Publishers for ultrasonic (for debugging)
        self.us_fl_pub = self.create_publisher(Range, 'ultrasonic/front_left', 10)
        self.us_fr_pub = self.create_publisher(Range, 'ultrasonic/front_right', 10)
        self.us_back_pub = self.create_publisher(Range, 'ultrasonic/back', 10)

        # Watchdog - stop if no commands received
        self.last_cmd_time = time.time()
        self.watchdog_timer = self.create_timer(0.5, self.watchdog_callback)

        # Ultrasonic polling timer (50Hz)
        self.us_timer = self.create_timer(0.02, self.poll_ultrasonic)

        # Lock for serial access
        self.serial_lock = threading.Lock()

        self.last_cmd = "STOP"

        self.get_logger().info('Arduino Bridge ready with LiDAR + Ultrasonic + Camera!')

    def camera_callback(self, msg: Float32MultiArray):
        """Process camera obstacle detection."""
        if len(msg.data) >= 3:
            self.cam_obstacles = list(msg.data[:3])
            # If camera detects obstacle in center, block front
            if self.cam_obstacles[1] > 0.7:  # Center obstacle score > 0.7
                self.front_clear = False

    def poll_ultrasonic(self):
        """Poll ultrasonic sensors from Arduino."""
        if self.arduino is None:
            return

        with self.serial_lock:
            try:
                self.arduino.write(b"ULTRASONIC\n")
                self.arduino.flush()
                line = self.arduino.readline().decode().strip()

                # Parse response: "US:fl,fr,back" (in cm)
                if line.startswith("US:"):
                    parts = line[3:].split(",")
                    if len(parts) >= 3:
                        fl_cm = float(parts[0])
                        fr_cm = float(parts[1])
                        back_cm = float(parts[2])

                        # Convert to meters (-1 stays as -1)
                        self.us_front_left = fl_cm / 100.0 if fl_cm >= 0 else -1.0
                        self.us_front_right = fr_cm / 100.0 if fr_cm >= 0 else -1.0
                        self.us_back = back_cm / 100.0 if back_cm >= 0 else -1.0

                        # Update clearance based on ultrasonic
                        self.update_clearance_from_ultrasonic()

            except Exception as e:
                pass  # Ignore read errors

    def update_clearance_from_ultrasonic(self):
        """Update obstacle clearance from ultrasonic sensors."""
        min_dist = self.min_obstacle_dist

        # Reset clearance flags first
        self.front_clear = True
        self.back_clear = True
        self.left_clear = True
        self.right_clear = True

        # Block if we have a valid reading that's too close
        if self.us_front_left >= 0 and self.us_front_left < min_dist:
            self.front_clear = False
            self.left_clear = False
        if self.us_front_right >= 0 and self.us_front_right < min_dist:
            self.front_clear = False
            self.right_clear = False
        if self.us_back >= 0 and self.us_back < min_dist:
            self.back_clear = False

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR scan for obstacle detection."""
        n_ranges = len(msg.ranges)
        if n_ranges == 0:
            return

        front_min = float('inf')
        back_min = float('inf')

        for i, r in enumerate(msg.ranges):
            if r < msg.range_min or r > msg.range_max or math.isnan(r) or math.isinf(r):
                continue

            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)

            while angle_deg > 180:
                angle_deg -= 360
            while angle_deg < -180:
                angle_deg += 360

            if -45 <= angle_deg <= 45:
                front_min = min(front_min, r)
            elif 135 <= angle_deg or angle_deg <= -135:
                back_min = min(back_min, r)

        # LiDAR also contributes to clearance (but ultrasonic takes priority)
        if front_min < self.min_obstacle_dist:
            self.front_clear = False
        if back_min < self.min_obstacle_dist:
            self.back_clear = False

    def cmd_vel_callback(self, msg: Twist):
        """Convert cmd_vel to Arduino motor commands WITH SAFETY."""
        self.last_cmd_time = time.time()

        linear = msg.linear.x
        angular = msg.angular.z

        # Log sensor state periodically
        if int(time.time()) % 5 == 0:
            self.get_logger().info(
                f'US: FL={self.us_front_left:.2f}m FR={self.us_front_right:.2f}m B={self.us_back:.2f}m | '
                f'Clear: F={self.front_clear} B={self.back_clear}'
            )

        # SAFETY: Check obstacles before moving
        if linear > 0.05 and not self.front_clear:
            self.get_logger().warn(
                f'BLOCKED FRONT: FL={self.us_front_left:.2f}m FR={self.us_front_right:.2f}m')
            self.send_command("STOP")
            return

        if linear < -0.05 and not self.back_clear:
            self.get_logger().warn(f'BLOCKED BACK: {self.us_back:.2f}m')
            self.send_command("STOP")
            return

        # Determine motor command
        if abs(linear) < 0.02 and abs(angular) < 0.05:
            cmd = "STOP"
        elif abs(angular) > abs(linear) * 1.5:
            cmd = "LEFT" if angular > 0 else "RIGHT"
        elif linear > 0:
            cmd = "FORWARD"
        else:
            cmd = "BACKWARD"

        self.send_command(cmd)

    def send_command(self, cmd: str):
        """Send command to Arduino."""
        if self.arduino is None:
            return

        # Don't spam same command
        if cmd == self.last_cmd and cmd == "STOP":
            return

        self.last_cmd = cmd

        with self.serial_lock:
            try:
                self.arduino.write(f"{cmd}\n".encode())
                self.arduino.flush()
            except Exception as e:
                self.get_logger().error(f'Serial write error: {e}')

    def watchdog_callback(self):
        """Stop robot if no commands received recently."""
        if time.time() - self.last_cmd_time > 1.0:
            self.send_command("STOP")
            # Reset clearance flags
            self.front_clear = True
            self.back_clear = True
            self.left_clear = True
            self.right_clear = True

    def destroy_node(self):
        if self.arduino:
            self.send_command("STOP")
            self.arduino.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArduinoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
