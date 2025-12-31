#!/usr/bin/env python3
"""
Debug Explorer - verbose output to find issues
"""

import time
import serial
import sys
from rplidar import RPLidar

# Force unbuffered output
sys.stdout = sys.stderr

print("=" * 50)
print("EXPLORER DEBUG MODE")
print("=" * 50)

# Motors
print("\n[1/3] Motors...")
arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2.5)
arduino.reset_input_buffer()

def motor(cmd):
    print(f"  -> Motor: {cmd}")
    arduino.write(f"{cmd}\n".encode())
    arduino.flush()
    time.sleep(0.05)
    # Read response
    if arduino.in_waiting:
        resp = arduino.readline().decode().strip()
        print(f"  <- Response: {resp}")

motor("STOP")
print("  Motors OK")

# LiDAR
print("\n[2/3] LiDAR...")
lidar = RPLidar('/dev/ttyUSB0')
lidar.stop()
lidar.stop_motor()
time.sleep(0.3)
lidar.start_motor()
time.sleep(1)
print("  LiDAR OK")

def get_distances(scan):
    """Get distances in front, left, right from LiDAR."""
    f, l, r = [], [], []
    for _, a, d in scan:
        if d <= 0:
            continue
        dm = d / 1000.0
        if a <= 30 or a >= 330:
            f.append(dm)
        elif 45 <= a <= 135:
            r.append(dm)
        elif 225 <= a <= 315:
            l.append(dm)

    front = min(f) if f else 9.0
    left = min(l) if l else 9.0
    right = min(r) if r else 9.0
    return front, left, right

print("\n[3/3] Starting exploration loop...")
print("-" * 50)
scan_iter = lidar.iter_scans(max_buf_meas=8000, min_len=5)
moves = 0

try:
    for i in range(20):  # Only 20 iterations for debug
        print(f"\n--- Move {i+1} ---")

        # Get scan
        print("Getting LiDAR scan...")
        try:
            scan = next(scan_iter)
            print(f"  Got {len(scan)} points")
        except Exception as e:
            print(f"  Scan error: {e}")
            continue

        front, left, right = get_distances(scan)
        print(f"  Distances: F={front:.2f}m L={left:.2f}m R={right:.2f}m")

        # Decision
        if front < 0.30:
            print("  Decision: TOO CLOSE - backup and turn")
            motor("BACKWARD")
            time.sleep(0.3)
            motor("STOP")
            if left > right:
                motor("LEFT")
            else:
                motor("RIGHT")
            time.sleep(0.4)
            motor("STOP")
        elif front < 0.50:
            print("  Decision: CLOSE - turning")
            if left > right:
                motor("LEFT")
            else:
                motor("RIGHT")
            time.sleep(0.3)
            motor("STOP")
        elif front < 0.80:
            print("  Decision: MEDIUM - short forward")
            motor("FORWARD")
            time.sleep(0.2)
            motor("STOP")
        else:
            print("  Decision: CLEAR - forward")
            motor("FORWARD")
            time.sleep(0.4)
            motor("STOP")

        moves += 1
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\nStopped by user")
except Exception as e:
    print(f"\n\nERROR: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
print("\n\nCleaning up...")
motor("STOP")

try:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
except:
    pass

arduino.close()
print(f"Done! Completed {moves} moves.")
