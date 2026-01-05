#!/usr/bin/env python3
"""
Motor Controller Node - Mecanum/Holonomic Drive
With velocity timeout and ultrasonic filtering.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Range
import serial
import time
from collections import deque
import statistics


class UltrasonicFilter:
    """Filter ultrasonic readings - remove outliers and average."""
    
    def __init__(self, window_size=5, outlier_threshold=2.0):
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold  # std devs from median
        self.buffer = deque(maxlen=window_size)
    
    def add_reading(self, value):
        """Add a new reading and return filtered value."""
        if value < 0:  # Invalid reading
            return None
        
        self.buffer.append(value)
        
        if len(self.buffer) < 3:
            return value  # Not enough data to filter
        
        # Get median
        median = statistics.median(self.buffer)
        
        # Filter outliers (values too far from median)
        valid_readings = []
        for reading in self.buffer:
            # Allow readings within 30% of median or within 20cm
            if abs(reading - median) <= max(median * 0.3, 20):
                valid_readings.append(reading)
        
        if not valid_readings:
            return median  # Fallback to median if all filtered
        
        # Return average of valid readings
        return sum(valid_readings) / len(valid_readings)


class MotorController(Node):
    def __init__(self):
        super().__init__("motor_controller")

        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("max_linear", 0.5)
        self.declare_parameter("max_angular", 2.0)
        self.declare_parameter("max_pwm", 180)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("ultrasonic_sensors", ["front_left", "front_right", "back_center"])
        self.declare_parameter("ultrasonic_filter_size", 5)

        self.serial_port = self.get_parameter("serial_port").value
        self.baud_rate = self.get_parameter("baud_rate").value
        self.max_linear = self.get_parameter("max_linear").value
        self.max_angular = self.get_parameter("max_angular").value
        self.max_pwm = self.get_parameter("max_pwm").value
        self.cmd_timeout = self.get_parameter("cmd_timeout").value
        self.ultrasonic_names = self.get_parameter("ultrasonic_sensors").value
        filter_size = self.get_parameter("ultrasonic_filter_size").value

        self.serial = None
        self.connect_serial()

        self.last_cmd_time = time.time()
        self.is_moving = False

        # Create ultrasonic filters
        self.ultrasonic_filters = {}
        for name in self.ultrasonic_names:
            self.ultrasonic_filters[name] = UltrasonicFilter(window_size=filter_size)

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
        self.create_timer(0.05, self.timer_callback)

        self.get_logger().info(f"Mecanum motor controller started on {self.serial_port}")
        self.get_logger().info(f"Ultrasonic filtering: window={filter_size}, outlier removal enabled")

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

        self.last_cmd_time = time.time()

        vx = msg.linear.x / self.max_linear
        vy = -msg.linear.y / self.max_linear
        wz = msg.angular.z / self.max_angular

        vx = max(-1.0, min(1.0, vx))
        vy = max(-1.0, min(1.0, vy))
        wz = max(-1.0, min(1.0, wz))

        lf = vx + vy + wz
        rf = vx - vy - wz
        lr = vx - vy + wz
        rr = vx + vy - wz

        max_val = max(abs(lf), abs(rf), abs(lr), abs(rr))
        if max_val > 1.0:
            lf /= max_val
            rf /= max_val
            lr /= max_val
            rr /= max_val

        lf_pwm = int(lf * self.max_pwm)
        rf_pwm = int(rf * self.max_pwm)
        lr_pwm = int(lr * self.max_pwm)
        rr_pwm = int(rr * self.max_pwm)

        self.is_moving = any([lf_pwm, rf_pwm, lr_pwm, rr_pwm])

        try:
            command = f"MECANUM,{lf_pwm},{rf_pwm},{lr_pwm},{rr_pwm}\n"
            self.serial.write(command.encode())
        except serial.SerialException as e:
            self.get_logger().error(f"Serial write error: {e}")

    def timer_callback(self):
        if self.serial is None:
            return

        if self.is_moving and (time.time() - self.last_cmd_time) > self.cmd_timeout:
            self.get_logger().info("Velocity timeout - stopping motors")
            try:
                self.serial.write(b"STOP\n")
                self.is_moving = False
            except serial.SerialException:
                pass

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
                    # Correct encoder signs: left negated, right as-is
                    msg.data = [-int(parts[0]), int(parts[1])]
                    self.encoder_pub.publish(msg)
            except ValueError:
                pass

        elif line.startswith("US:"):
            try:
                parts = line[3:].split(",")
                if len(parts) >= 3:
                    raw_distances = [float(p) for p in parts]
                    now = self.get_clock().now().to_msg()

                    for i, name in enumerate(self.ultrasonic_names):
                        if i < len(raw_distances):
                            raw_cm = raw_distances[i]
                            
                            # Apply filter
                            filtered_cm = self.ultrasonic_filters[name].add_reading(raw_cm)
                            
                            if filtered_cm is None:
                                continue
                            
                            msg = Range()
                            msg.header.stamp = now
                            msg.header.frame_id = f"ultrasonic_{name}_link"
                            msg.radiation_type = Range.ULTRASOUND
                            msg.field_of_view = 0.26
                            msg.min_range = 0.02
                            msg.max_range = 4.0
                            msg.range = filtered_cm / 100.0  # Convert to meters

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
