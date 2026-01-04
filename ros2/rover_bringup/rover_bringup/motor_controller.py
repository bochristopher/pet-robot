#!/usr/bin/env python3
"""
Motor Controller Node - Mecanum/Holonomic Drive
Sends MECANUM command with 4 wheel PWM values.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Range
import serial
import time


class MotorController(Node):
    def __init__(self):
        super().__init__("motor_controller")

        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("max_linear", 0.5)
        self.declare_parameter("max_angular", 2.0)
        self.declare_parameter("max_pwm", 180)
        self.declare_parameter("ultrasonic_sensors", ["front_left", "front_right", "back_center"])

        self.serial_port = self.get_parameter("serial_port").value
        self.baud_rate = self.get_parameter("baud_rate").value
        self.max_linear = self.get_parameter("max_linear").value
        self.max_angular = self.get_parameter("max_angular").value
        self.max_pwm = self.get_parameter("max_pwm").value
        self.ultrasonic_names = self.get_parameter("ultrasonic_sensors").value

        self.serial = None
        self.connect_serial()

        self.cmd_vel_sub = self.create_subscription(
            Twist, "cmd_vel", self.cmd_vel_callback, 10
        )

        self.encoder_pub = self.create_publisher(Int32MultiArray, "wheel_encoders", 10)

        self.ultrasonic_pubs = {}
        for name in self.ultrasonic_names:
            self.ultrasonic_pubs[name] = self.create_publisher(
                Range, f"ultrasonic/{name}", 10
            )

        self.read_state = 0
        self.create_timer(0.05, self.read_sensors)

        self.get_logger().info(f"Mecanum motor controller started on {self.serial_port}")

    def connect_serial(self):
        try:
            self.serial = serial.Serial(
                self.serial_port, self.baud_rate, timeout=0.1
            )
            time.sleep(2.0)
            while self.serial.in_waiting:
                self.serial.readline()
            self.get_logger().info(f"Connected to Arduino on {self.serial_port}")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to connect: {e}")
            self.serial = None

    def cmd_vel_callback(self, msg: Twist):
        if self.serial is None:
            return

        # Get velocities
        vx = msg.linear.x / self.max_linear   # forward/backward normalized
        vy = -msg.linear.y / self.max_linear   # strafe normalized
        wz = msg.angular.z / self.max_angular # rotation normalized

        # Clamp to -1 to 1
        vx = max(-1.0, min(1.0, vx))
        vy = max(-1.0, min(1.0, vy))
        wz = max(-1.0, min(1.0, wz))

        # Mecanum kinematics: compute wheel speeds
        # LF = vx + vy + wz
        # RF = vx - vy - wz
        # LR = vx - vy + wz
        # RR = vx + vy - wz
        lf = vx + vy + wz
        rf = vx - vy - wz
        lr = vx - vy + wz
        rr = vx + vy - wz

        # Normalize if any exceeds 1.0
        max_val = max(abs(lf), abs(rf), abs(lr), abs(rr))
        if max_val > 1.0:
            lf /= max_val
            rf /= max_val
            lr /= max_val
            rr /= max_val

        # Convert to PWM
        lf_pwm = int(lf * self.max_pwm)
        rf_pwm = int(rf * self.max_pwm)
        lr_pwm = int(lr * self.max_pwm)
        rr_pwm = int(rr * self.max_pwm)

        try:
            command = f"MECANUM,{lf_pwm},{rf_pwm},{lr_pwm},{rr_pwm}\n"
            self.serial.write(command.encode())
        except serial.SerialException as e:
            self.get_logger().error(f"Serial write error: {e}")

    def read_sensors(self):
        if self.serial is None:
            return

        try:
            if self.read_state == 0:
                self.serial.write(b"ENCODER\n")
            else:
                self.serial.write(b"ULTRASONIC\n")

            self.read_state = (self.read_state + 1) % 2

            while self.serial.in_waiting > 0:
                line = self.serial.readline().decode().strip()
                self.process_response(line)

        except (serial.SerialException, ValueError, UnicodeDecodeError) as e:
            self.get_logger().debug(f"Serial read error: {e}")

    def process_response(self, line: str):
        if not line:
            return

        if line.startswith("ENC:"):
            try:
                parts = line[4:].split(",")
                if len(parts) == 2:
                    msg = Int32MultiArray()
                    msg.data = [int(parts[0]), int(parts[1])]
                    self.encoder_pub.publish(msg)
            except ValueError:
                pass

        elif line.startswith("US:"):
            try:
                parts = line[3:].split(",")
                if len(parts) >= 3:
                    distances = [float(p) for p in parts]
                    now = self.get_clock().now().to_msg()

                    for i, name in enumerate(self.ultrasonic_names):
                        if i < len(distances):
                            msg = Range()
                            msg.header.stamp = now
                            msg.header.frame_id = f"ultrasonic_{name}_link"
                            msg.radiation_type = Range.ULTRASOUND
                            msg.field_of_view = 0.26
                            msg.min_range = 0.02
                            msg.max_range = 4.0

                            dist_cm = distances[i]
                            if dist_cm < 0:
                                msg.range = float("inf")
                            else:
                                msg.range = dist_cm / 100.0

                            self.ultrasonic_pubs[name].publish(msg)
            except ValueError:
                pass

    def destroy_node(self):
        if self.serial:
            try:
                self.serial.write(b"STOP\n")
                self.serial.close()
            except:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
