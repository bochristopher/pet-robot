#!/usr/bin/env python3
"""
MPU-6050 IMU Module for Jetson
Provides accelerometer, gyroscope, and computed angles.
"""

import smbus2
import time
import math
from dataclasses import dataclass
from typing import Tuple, Optional

# MPU-6050 Registers
MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
WHO_AM_I = 0x75

# Scale factors
ACCEL_SCALE_2G = 16384.0
GYRO_SCALE_250DPS = 131.0


@dataclass
class IMUData:
    """IMU sensor readings."""
    # Raw accelerometer (g)
    accel_x: float
    accel_y: float
    accel_z: float
    # Raw gyroscope (deg/s)
    gyro_x: float
    gyro_y: float
    gyro_z: float
    # Computed angles (degrees)
    roll: float
    pitch: float
    yaw: float
    # Timestamp
    timestamp: float


class MPU6050:
    """MPU-6050 6-axis IMU driver."""

    def __init__(self, bus_num: int = 7, address: int = MPU_ADDR):
        self.bus_num = bus_num
        self.address = address
        self.bus = None

        # Calibration offsets
        self.gyro_offset_x = 0.0
        self.gyro_offset_y = 0.0
        self.gyro_offset_z = 0.0
        self.accel_offset_x = 0.0
        self.accel_offset_y = 0.0
        self.accel_offset_z = 0.0

        # Complementary filter state
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_time = None

        # Filter coefficient (0.0-1.0, higher = trust gyro more)
        self.alpha = 0.96

        self._init_sensor()

    def _init_sensor(self):
        """Initialize the MPU-6050."""
        self.bus = smbus2.SMBus(self.bus_num)

        # Check WHO_AM_I
        who = self.bus.read_byte_data(self.address, WHO_AM_I)
        if who != 0x68:
            raise RuntimeError(f"MPU-6050 not found. WHO_AM_I=0x{who:02x}")

        # Wake up (clear sleep bit)
        self.bus.write_byte_data(self.address, PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # Set sample rate divider (1kHz / (1 + 4) = 200Hz)
        self.bus.write_byte_data(self.address, SMPLRT_DIV, 4)

        # Set DLPF (Digital Low Pass Filter) to 44Hz
        self.bus.write_byte_data(self.address, CONFIG, 0x03)

        # Set gyro range to +/- 250 deg/s
        self.bus.write_byte_data(self.address, GYRO_CONFIG, 0x00)

        # Set accel range to +/- 2g
        self.bus.write_byte_data(self.address, ACCEL_CONFIG, 0x00)

        time.sleep(0.1)
        print(f"[IMU] MPU-6050 initialized on I2C bus {self.bus_num}")

    def _read_word(self, reg: int) -> int:
        """Read a signed 16-bit value from two registers."""
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        val = (high << 8) + low
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val

    def _read_raw(self) -> Tuple[int, int, int, int, int, int]:
        """Read raw sensor values."""
        ax = self._read_word(ACCEL_XOUT_H)
        ay = self._read_word(ACCEL_XOUT_H + 2)
        az = self._read_word(ACCEL_XOUT_H + 4)
        gx = self._read_word(GYRO_XOUT_H)
        gy = self._read_word(GYRO_XOUT_H + 2)
        gz = self._read_word(GYRO_XOUT_H + 4)
        return ax, ay, az, gx, gy, gz

    def calibrate(self, samples: int = 100):
        """Calibrate gyro offsets. Keep sensor still!"""
        print("[IMU] Calibrating... keep sensor still!")

        gx_sum = gy_sum = gz_sum = 0.0
        ax_sum = ay_sum = az_sum = 0.0

        for _ in range(samples):
            ax, ay, az, gx, gy, gz = self._read_raw()
            ax_sum += ax / ACCEL_SCALE_2G
            ay_sum += ay / ACCEL_SCALE_2G
            az_sum += az / ACCEL_SCALE_2G
            gx_sum += gx / GYRO_SCALE_250DPS
            gy_sum += gy / GYRO_SCALE_250DPS
            gz_sum += gz / GYRO_SCALE_250DPS
            time.sleep(0.01)

        self.gyro_offset_x = gx_sum / samples
        self.gyro_offset_y = gy_sum / samples
        self.gyro_offset_z = gz_sum / samples

        # Accel offsets (assuming Z should be 1g when flat)
        self.accel_offset_x = ax_sum / samples
        self.accel_offset_y = ay_sum / samples
        self.accel_offset_z = (az_sum / samples) - 1.0  # Subtract 1g

        # Reset angles
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_time = None

        print(f"[IMU] Calibration complete")
        print(f"      Gyro offsets: X={self.gyro_offset_x:.2f} Y={self.gyro_offset_y:.2f} Z={self.gyro_offset_z:.2f}")

    def read(self) -> IMUData:
        """Read sensor and compute angles using complementary filter."""
        now = time.time()
        ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw = self._read_raw()

        # Convert to physical units
        ax = (ax_raw / ACCEL_SCALE_2G) - self.accel_offset_x
        ay = (ay_raw / ACCEL_SCALE_2G) - self.accel_offset_y
        az = (az_raw / ACCEL_SCALE_2G) - self.accel_offset_z

        gx = (gx_raw / GYRO_SCALE_250DPS) - self.gyro_offset_x
        gy = (gy_raw / GYRO_SCALE_250DPS) - self.gyro_offset_y
        gz = (gz_raw / GYRO_SCALE_250DPS) - self.gyro_offset_z

        # Compute angles from accelerometer
        accel_roll = math.atan2(ay, az) * 180.0 / math.pi
        accel_pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az)) * 180.0 / math.pi

        # Complementary filter
        if self.last_time is None:
            # First reading - use accelerometer angles
            self.roll = accel_roll
            self.pitch = accel_pitch
            self.yaw = 0.0
        else:
            dt = now - self.last_time
            # Integrate gyro and fuse with accelerometer
            self.roll = self.alpha * (self.roll + gx * dt) + (1 - self.alpha) * accel_roll
            self.pitch = self.alpha * (self.pitch + gy * dt) + (1 - self.alpha) * accel_pitch
            self.yaw += gz * dt  # Yaw from gyro only (no magnetometer)

        self.last_time = now

        return IMUData(
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
            timestamp=now
        )

    def get_heading(self) -> float:
        """Get current yaw/heading in degrees."""
        return self.yaw

    def reset_yaw(self):
        """Reset yaw to zero."""
        self.yaw = 0.0

    def close(self):
        """Close I2C connection."""
        if self.bus:
            self.bus.close()
            self.bus = None


# Singleton instance
_imu: Optional[MPU6050] = None


def get_imu() -> MPU6050:
    """Get or create the global IMU instance."""
    global _imu
    if _imu is None:
        _imu = MPU6050()
    return _imu


if __name__ == "__main__":
    # Test the IMU
    imu = MPU6050()
    imu.calibrate(50)

    print("\nReading IMU data (Ctrl+C to stop):")
    print("-" * 60)

    try:
        while True:
            data = imu.read()
            print(f"\rRoll:{data.roll:7.2f}  Pitch:{data.pitch:7.2f}  Yaw:{data.yaw:7.2f}  "
                  f"Gyro Z:{data.gyro_z:6.1f}    ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        imu.close()
