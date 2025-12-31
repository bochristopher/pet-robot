#!/usr/bin/env python3
"""
Encoder Interface Module
Communicates with Arduino to read wheel encoder data.
"""

import serial
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

# Default settings
DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200

# Wheel parameters (adjust for your robot)
WHEEL_DIAMETER_MM = 80.0  # mm
ENCODER_CPR = 360  # Counts per revolution (adjust for your encoder)
WHEEL_BASE_MM = 200.0  # Distance between wheels (mm)


@dataclass
class EncoderData:
    """Encoder readings."""
    left_count: int
    right_count: int
    left_speed: float  # counts/sec
    right_speed: float  # counts/sec
    timestamp: float


@dataclass
class Odometry:
    """Robot odometry state."""
    x: float  # mm
    y: float  # mm
    theta: float  # radians
    velocity: float  # mm/s
    angular_velocity: float  # rad/s
    timestamp: float


class EncoderInterface:
    """Interface to Arduino motor controller with encoders."""

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD, serial_conn=None):
        """
        Initialize encoder interface.

        Args:
            port: Serial port (ignored if serial_conn provided)
            baud: Baud rate (ignored if serial_conn provided)
            serial_conn: Optional existing serial.Serial connection to share
        """
        self.port = port
        self.baud = baud
        self.serial = serial_conn  # Use existing connection if provided
        self._own_serial = serial_conn is None  # Track if we created it
        self.connected = False

        # Encoder state
        self.left_count = 0
        self.right_count = 0
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.last_left = 0
        self.last_right = 0
        self.last_time = None

        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Wheel parameters
        self.wheel_circumference = WHEEL_DIAMETER_MM * 3.14159
        self.mm_per_count = self.wheel_circumference / ENCODER_CPR
        self.wheel_base = WHEEL_BASE_MM

        # Thread for continuous reading
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self._connect()

    def _connect(self) -> bool:
        """Connect to Arduino."""
        try:
            # If serial was provided externally, just use it
            if self.serial is not None:
                self.connected = True
                print(f"[Encoder] Using shared serial connection")
                return True

            # Otherwise create our own connection
            self.serial = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # Wait for Arduino reset

            # Clear buffer
            self.serial.reset_input_buffer()

            # Read welcome message
            for _ in range(5):
                if self.serial.in_waiting:
                    line = self.serial.readline().decode().strip()
                    if "Ready" in line:
                        self.connected = True
                        print(f"[Encoder] Connected to Arduino on {self.port}")
                        return True
                time.sleep(0.2)

            self.connected = True  # Assume connected anyway
            print(f"[Encoder] Connected to {self.port}")
            return True

        except Exception as e:
            print(f"[Encoder] Connection failed: {e}")
            self.connected = False
            return False

    def _send_command(self, cmd: str) -> Optional[str]:
        """Send command and get response."""
        if not self.connected:
            return None
        try:
            self.serial.write(f"{cmd}\n".encode())
            time.sleep(0.05)
            if self.serial.in_waiting:
                return self.serial.readline().decode().strip()
        except Exception as e:
            print(f"[Encoder] Command error: {e}")
        return None

    def read_encoders(self) -> Optional[EncoderData]:
        """Read current encoder counts."""
        response = self._send_command("ENCODERS")
        if response and response.startswith("ENC:"):
            try:
                parts = response[4:].split(",")
                left = int(parts[0])
                right = int(parts[1])

                now = time.time()
                with self._lock:
                    self.left_count = left
                    self.right_count = right

                    # Calculate speed
                    if self.last_time:
                        dt = now - self.last_time
                        if dt > 0:
                            self.left_speed = (left - self.last_left) / dt
                            self.right_speed = (right - self.last_right) / dt

                    self.last_left = left
                    self.last_right = right
                    self.last_time = now

                return EncoderData(
                    left_count=left,
                    right_count=right,
                    left_speed=self.left_speed,
                    right_speed=self.right_speed,
                    timestamp=now
                )
            except (ValueError, IndexError):
                pass
        return None

    def read_speed(self) -> Optional[Tuple[float, float]]:
        """Read speed from Arduino."""
        response = self._send_command("SPEED")
        if response and response.startswith("SPD:"):
            try:
                parts = response[4:].split(",")
                return float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                pass
        return None

    def reset_encoders(self):
        """Reset encoder counts to zero."""
        self._send_command("RESET")
        with self._lock:
            self.left_count = 0
            self.right_count = 0
            self.last_left = 0
            self.last_right = 0
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0
        print("[Encoder] Counts reset")

    def update_odometry(self) -> Odometry:
        """Update odometry from encoder readings."""
        data = self.read_encoders()
        if not data:
            return Odometry(self.x, self.y, self.theta, 0, 0, time.time())

        with self._lock:
            # Calculate distance traveled by each wheel
            left_dist = data.left_speed * self.mm_per_count
            right_dist = data.right_speed * self.mm_per_count

            # Calculate linear and angular velocity
            velocity = (left_dist + right_dist) / 2.0
            angular_vel = (right_dist - left_dist) / self.wheel_base

            # Update position (simple dead reckoning)
            if self.last_time:
                dt = data.timestamp - self.last_time
                self.x += velocity * dt * math.cos(self.theta)
                self.y += velocity * dt * math.sin(self.theta)
                self.theta += angular_vel * dt

            return Odometry(
                x=self.x, y=self.y, theta=self.theta,
                velocity=velocity, angular_velocity=angular_vel,
                timestamp=data.timestamp
            )

    def get_counts(self) -> Tuple[int, int]:
        """Get current encoder counts."""
        with self._lock:
            return self.left_count, self.right_count

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position (x, y, theta)."""
        with self._lock:
            return self.x, self.y, self.theta

    # Motor control commands
    def forward(self):
        self._send_command("FORWARD")

    def backward(self):
        self._send_command("BACKWARD")

    def left(self):
        self._send_command("LEFT")

    def right(self):
        self._send_command("RIGHT")

    def stop(self):
        self._send_command("STOP")

    def close(self):
        """Close serial connection (only if we created it)."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        # Only close serial if we own it (created it ourselves)
        if self.serial and self._own_serial:
            self.serial.close()
            self.serial = None
        self.connected = False
        print("[Encoder] Disconnected")


# Need math for odometry
import math

# Singleton
_encoders: Optional[EncoderInterface] = None


def get_encoders() -> EncoderInterface:
    """Get or create the global encoder interface."""
    global _encoders
    if _encoders is None:
        _encoders = EncoderInterface()
    return _encoders


if __name__ == "__main__":
    # Test encoder interface
    enc = EncoderInterface()

    if not enc.connected:
        print("Failed to connect to Arduino")
        exit(1)

    print("\nReading encoders (Ctrl+C to stop):")
    print("-" * 50)

    try:
        while True:
            data = enc.read_encoders()
            if data:
                print(f"\rL:{data.left_count:6d} R:{data.right_count:6d}  "
                      f"Speed L:{data.left_speed:6.1f} R:{data.right_speed:6.1f}    ",
                      end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        enc.close()
