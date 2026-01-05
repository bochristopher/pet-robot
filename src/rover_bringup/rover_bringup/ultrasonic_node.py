#!/usr/bin/env python3
"""
Ultrasonic Sensor Node
Reads HC-SR04 ultrasonic sensors via GPIO and publishes distance data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
import time

# Defer GPIO import to avoid crash if Jetson model detection fails
GPIO = None
GPIO_AVAILABLE = False


class UltrasonicNode(Node):
    def __init__(self):
        super().__init__('ultrasonic_node')

        # Try to import Jetson.GPIO
        global GPIO, GPIO_AVAILABLE
        try:
            import Jetson.GPIO as jetson_gpio
            GPIO = jetson_gpio
            GPIO_AVAILABLE = True
            self.get_logger().info('Jetson.GPIO loaded successfully')
        except Exception as e:
            self.get_logger().warn(f'Jetson.GPIO not available: {e}. Using simulated data.')
            GPIO_AVAILABLE = False

        # Parameters - GPIO pin pairs (trigger, echo)
        self.declare_parameter('sensors', [
            {'name': 'front', 'trigger': 17, 'echo': 27},
            {'name': 'left', 'trigger': 22, 'echo': 10},
        ])
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('max_range', 4.0)  # meters
        self.declare_parameter('min_range', 0.02)  # meters
        self.declare_parameter('field_of_view', 0.26)  # ~15 degrees in radians

        self.max_range = self.get_parameter('max_range').value
        self.min_range = self.get_parameter('min_range').value
        self.fov = self.get_parameter('field_of_view').value
        rate = self.get_parameter('publish_rate').value

        # Sensor configurations
        self.sensors = [
            {'name': 'front', 'trigger': 17, 'echo': 27},
            {'name': 'left', 'trigger': 22, 'echo': 10},
        ]

        # Publishers for each sensor
        self.publishers = {}
        for sensor in self.sensors:
            topic = f"ultrasonic/{sensor['name']}"
            self.publishers[sensor['name']] = self.create_publisher(Range, topic, 10)

        # Setup GPIO
        if GPIO_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                for sensor in self.sensors:
                    GPIO.setup(sensor['trigger'], GPIO.OUT, initial=GPIO.LOW)
                    GPIO.setup(sensor['echo'], GPIO.IN)
                self.get_logger().info('GPIO initialized for ultrasonic sensors')
            except Exception as e:
                self.get_logger().error(f'GPIO setup failed: {e}')
        else:
            self.get_logger().warn('Jetson.GPIO not available, using simulated data')

        # Timer
        self.sensor_index = 0
        self.create_timer(1.0 / rate, self.read_and_publish)

    def measure_distance(self, trigger_pin, echo_pin):
        """Measure distance using HC-SR04 sensor."""
        if not GPIO_AVAILABLE:
            return 1.0  # Simulated distance

        try:
            # Send trigger pulse
            GPIO.output(trigger_pin, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(trigger_pin, GPIO.LOW)

            # Wait for echo start
            pulse_start = time.time()
            timeout = pulse_start + 0.1
            while GPIO.input(echo_pin) == GPIO.LOW:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return -1.0

            # Wait for echo end
            pulse_end = time.time()
            timeout = pulse_end + 0.1
            while GPIO.input(echo_pin) == GPIO.HIGH:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return -1.0

            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 343.0 / 2.0  # Speed of sound / 2

            return distance

        except Exception as e:
            self.get_logger().debug(f'Measurement error: {e}')
            return -1.0

    def read_and_publish(self):
        """Read one sensor per cycle (round-robin)."""
        sensor = self.sensors[self.sensor_index]
        self.sensor_index = (self.sensor_index + 1) % len(self.sensors)

        distance = self.measure_distance(sensor['trigger'], sensor['echo'])

        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"ultrasonic_{sensor['name']}_link"
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = self.fov
        msg.min_range = self.min_range
        msg.max_range = self.max_range

        if distance < 0:
            msg.range = float('inf')  # Invalid reading
        else:
            msg.range = max(self.min_range, min(self.max_range, distance))

        self.publishers[sensor['name']].publish(msg)

    def destroy_node(self):
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
            except:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
