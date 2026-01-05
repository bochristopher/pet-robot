#!/usr/bin/env python3
"""
IMU Node for MPU6050
Reads accelerometer and gyroscope data via I2C and publishes to /imu/data
Includes complementary filter for orientation estimation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import smbus2
import math
import time


def euler_to_quaternion(roll, pitch, yaw):
    """Convert euler angles (radians) to quaternion."""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


class MPU6050:
    """MPU6050 I2C driver with complementary filter."""

    # MPU6050 Registers
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    # Scale factors
    ACCEL_SCALE = 16384.0  # LSB/g for ±2g
    GYRO_SCALE = 131.0     # LSB/(°/s) for ±250°/s

    def __init__(self, bus_num=1, address=0x68):
        self.bus = smbus2.SMBus(bus_num)
        self.address = address

        # Complementary filter state
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_time = None
        self.alpha = 0.96  # Trust gyro 96%, accel 4%

        # Calibration offsets
        self.gyro_offset_x = 0.0
        self.gyro_offset_y = 0.0
        self.gyro_offset_z = 0.0
        self.accel_offset_x = 0.0
        self.accel_offset_y = 0.0
        self.accel_offset_z = 0.0
        self.accel_scale = 1.0

        self._init_device()
        self._calibrate_gyro()
        self._calibrate_accel()

    def _init_device(self):
        """Initialize MPU6050."""
        # Wake up the device
        self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # Set sample rate to 100Hz
        self.bus.write_byte_data(self.address, self.SMPLRT_DIV, 0x09)

        # Set DLPF to 44Hz
        self.bus.write_byte_data(self.address, self.CONFIG, 0x03)

        # Set gyro range to ±250°/s
        self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)

        # Set accel range to ±2g
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)

    def _calibrate_gyro(self, samples=50):
        """Calibrate gyro offsets."""
        gx_sum = gy_sum = gz_sum = 0.0
        for _ in range(samples):
            gx, gy, gz = self._read_gyro_raw()
            gx_sum += gx
            gy_sum += gy
            gz_sum += gz
            time.sleep(0.01)
        self.gyro_offset_x = gx_sum / samples
        self.gyro_offset_y = gy_sum / samples
        self.gyro_offset_z = gz_sum / samples

    def _calibrate_accel(self, samples=50):
        """Calibrate accelerometer offsets. Assumes sensor is level and stationary."""
        ax_sum = ay_sum = az_sum = 0.0
        for _ in range(samples):
            ax, ay, az = self._read_accel_raw()
            ax_sum += ax
            ay_sum += ay
            az_sum += az
            time.sleep(0.01)
        # X and Y should be 0 when level
        self.accel_offset_x = ax_sum / samples
        self.accel_offset_y = ay_sum / samples
        # Z should be 9.81 when level, calculate scale factor
        avg_z = az_sum / samples
        self.accel_scale = 9.81 / avg_z if abs(avg_z) > 0.1 else 1.0
        # Don't offset Z, just scale it
        self.accel_offset_z = 0.0

    def _read_word(self, reg):
        """Read a 16-bit signed value."""
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        value = (high << 8) | low
        if value >= 0x8000:
            value = -((65535 - value) + 1)
        return value

    def _read_gyro_raw(self):
        """Read raw gyroscope data in rad/s."""
        gx = self._read_word(self.GYRO_XOUT_H) / self.GYRO_SCALE * (math.pi / 180.0)
        gy = self._read_word(self.GYRO_XOUT_H + 2) / self.GYRO_SCALE * (math.pi / 180.0)
        gz = self._read_word(self.GYRO_XOUT_H + 4) / self.GYRO_SCALE * (math.pi / 180.0)
        return gx, gy, gz

    def _read_accel_raw(self):
        """Read raw accelerometer data in m/s² (uncalibrated)."""
        ax = self._read_word(self.ACCEL_XOUT_H) / self.ACCEL_SCALE * 9.81
        ay = self._read_word(self.ACCEL_XOUT_H + 2) / self.ACCEL_SCALE * 9.81
        az = self._read_word(self.ACCEL_XOUT_H + 4) / self.ACCEL_SCALE * 9.81
        return ax, ay, az

    def read_accel(self):
        """Read calibrated accelerometer data in m/s²."""
        ax, ay, az = self._read_accel_raw()
        ax = (ax - self.accel_offset_x) * self.accel_scale
        ay = (ay - self.accel_offset_y) * self.accel_scale
        az = az * self.accel_scale
        return ax, ay, az

    def read_gyro(self):
        """Read calibrated gyroscope data in rad/s."""
        gx, gy, gz = self._read_gyro_raw()
        gx -= self.gyro_offset_x
        gy -= self.gyro_offset_y
        gz -= self.gyro_offset_z
        return gx, gy, gz

    def update_orientation(self):
        """Update roll/pitch/yaw using complementary filter."""
        now = time.time()
        ax, ay, az = self.read_accel()
        gx, gy, gz = self.read_gyro()

        # Compute angles from accelerometer
        accel_roll = math.atan2(ay, az)
        accel_pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))

        if self.last_time is None:
            # First reading
            self.roll = accel_roll
            self.pitch = accel_pitch
            self.yaw = 0.0
        else:
            dt = now - self.last_time
            # Complementary filter
            self.roll = self.alpha * (self.roll + gx * dt) + (1 - self.alpha) * accel_roll
            self.pitch = self.alpha * (self.pitch + gy * dt) + (1 - self.alpha) * accel_pitch
            self.yaw += gz * dt  # Yaw from gyro only

        self.last_time = now
        return ax, ay, az, gx, gy, gz

    def get_orientation(self):
        """Get current orientation as quaternion."""
        return euler_to_quaternion(self.roll, self.pitch, self.yaw)

    def close(self):
        self.bus.close()


class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')

        # Parameters
        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_address', 0x68)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_rate', 100.0)

        bus_num = self.get_parameter('i2c_bus').value
        address = self.get_parameter('i2c_address').value
        self.frame_id = self.get_parameter('frame_id').value
        rate = self.get_parameter('publish_rate').value

        # Initialize MPU6050
        self.mpu = None
        try:
            self.mpu = MPU6050(bus_num, address)
            self.get_logger().info(f'MPU6050 initialized on I2C bus {bus_num}, address 0x{address:02x}')
            self.get_logger().info(f'Accel calibration: offset=({self.mpu.accel_offset_x:.3f}, {self.mpu.accel_offset_y:.3f}), scale={self.mpu.accel_scale:.4f}')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize MPU6050: {e}')

        # Publisher
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)

        # Timer
        self.create_timer(1.0 / rate, self.publish_imu)

    def publish_imu(self):
        if self.mpu is None:
            return

        try:
            # Update orientation and get sensor data
            ax, ay, az, gx, gy, gz = self.mpu.update_orientation()
            qx, qy, qz, qw = self.mpu.get_orientation()

            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            # Orientation (quaternion)
            msg.orientation.x = qx
            msg.orientation.y = qy
            msg.orientation.z = qz
            msg.orientation.w = qw

            # Angular velocity (rad/s)
            msg.angular_velocity.x = gx
            msg.angular_velocity.y = gy
            msg.angular_velocity.z = gz

            # Linear acceleration (m/s²)
            msg.linear_acceleration.x = ax
            msg.linear_acceleration.y = ay
            msg.linear_acceleration.z = az

            # Covariances
            msg.orientation_covariance[0] = 0.01
            msg.orientation_covariance[4] = 0.01
            msg.orientation_covariance[8] = 0.01

            msg.angular_velocity_covariance[0] = 0.02
            msg.angular_velocity_covariance[4] = 0.02
            msg.angular_velocity_covariance[8] = 0.02

            msg.linear_acceleration_covariance[0] = 0.04
            msg.linear_acceleration_covariance[4] = 0.04
            msg.linear_acceleration_covariance[8] = 0.04

            self.imu_pub.publish(msg)

        except Exception as e:
            self.get_logger().debug(f'IMU read error: {e}')

    def destroy_node(self):
        if self.mpu:
            self.mpu.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
