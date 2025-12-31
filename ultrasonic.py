#!/usr/bin/env python3
"""
HC-SR04 Ultrasonic Sensor Driver via Arduino

Sensors are connected to Arduino Mega and read via serial command.
Arduino pin mapping:
- Front Left:  Trig=48, Echo=49
- Front Right: Trig=52, Echo=53
- Back:        Trig=50, Echo=51

Command: ULTRASONIC
Response: US:FL,FR,BACK (values in cm, -1 for timeout)
"""

import time
import serial
from dataclasses import dataclass
from typing import Optional


@dataclass
class UltrasonicReading:
    front_left: float   # cm (-1 = timeout)
    front_right: float  # cm
    back: float         # cm


class UltrasonicSensors:
    """Interface for 3 HC-SR04 ultrasonic sensors via Arduino."""

    def __init__(self, serial_conn: Optional[serial.Serial] = None, port: str = '/dev/ttyACM0'):
        """
        Initialize ultrasonic sensor interface.

        Args:
            serial_conn: Existing serial connection to share with other code
            port: Serial port if creating new connection
        """
        print("[Ultrasonic] Initializing via Arduino...")

        self._owns_serial = serial_conn is None
        if serial_conn:
            self.serial = serial_conn
        else:
            self.serial = serial.Serial(port, 115200, timeout=1)
            time.sleep(2.0)  # Wait for Arduino reset
            # Flush startup messages
            while self.serial.in_waiting:
                self.serial.readline()

        print("  Front Left:  Arduino pins 48/49")
        print("  Front Right: Arduino pins 52/53")
        print("  Back:        Arduino pins 50/51")
        print("[Ultrasonic] Ready")

    def read_all(self) -> UltrasonicReading:
        """Read all three sensors."""
        # Flush any pending data
        while self.serial.in_waiting:
            self.serial.readline()

        # Send command
        self.serial.write(b"ULTRASONIC\n")
        self.serial.flush()

        # Read response (try up to 3 times in case of other messages)
        for _ in range(3):
            try:
                response = self.serial.readline().decode().strip()
                if response.startswith("US:"):
                    parts = response[3:].split(",")
                    if len(parts) == 3:
                        fl = float(parts[0])
                        fr = float(parts[1])
                        back = float(parts[2])
                        return UltrasonicReading(front_left=fl, front_right=fr, back=back)
            except (ValueError, UnicodeDecodeError):
                pass

        return UltrasonicReading(front_left=-1, front_right=-1, back=-1)

    def read_front_left(self) -> float:
        """Read front left sensor."""
        reading = self.read_all()
        return reading.front_left

    def read_front_right(self) -> float:
        """Read front right sensor."""
        reading = self.read_all()
        return reading.front_right

    def read_back(self) -> float:
        """Read back sensor."""
        reading = self.read_all()
        return reading.back

    def close(self):
        """Close serial connection if we own it."""
        if self._owns_serial and self.serial:
            self.serial.close()
        print("[Ultrasonic] Closed")


def test_sensors():
    """Test all ultrasonic sensors."""
    print("=" * 50)
    print("ULTRASONIC SENSOR TEST (via Arduino)")
    print("=" * 50)

    try:
        sensors = UltrasonicSensors()
    except serial.SerialException as e:
        print(f"Error: Could not connect to Arduino: {e}")
        return

    print("\nTaking 10 readings (Ctrl+C to stop)...")
    try:
        for i in range(10):
            reading = sensors.read_all()
            fl = f"{reading.front_left:.0f}cm" if reading.front_left >= 0 else "timeout"
            fr = f"{reading.front_right:.0f}cm" if reading.front_right >= 0 else "timeout"
            bk = f"{reading.back:.0f}cm" if reading.back >= 0 else "timeout"
            print(f"  {i+1:2d}. FL: {fl:>10}  FR: {fr:>10}  Back: {bk:>10}")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nStopped by user")

    sensors.close()
    print("\nDone!")


if __name__ == "__main__":
    test_sensors()
