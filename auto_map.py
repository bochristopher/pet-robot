#!/usr/bin/env python3
"""
Floor Mapper with Obstacle Avoidance
Uses LiDAR + Ultrasonic sensors for safety
"""

import time
import math
import numpy as np
import cv2
import serial
import os
from datetime import datetime
from rplidar import RPLidar

# Try to import ultrasonic sensors
try:
    from ultrasonic import UltrasonicSensors
    ULTRASONIC_AVAILABLE = True
except:
    ULTRASONIC_AVAILABLE = False

# Try to import navigation (encoder + IMU fusion)
try:
    from navigation import Navigation
    NAVIGATION_AVAILABLE = True
except:
    NAVIGATION_AVAILABLE = False

MAP_SIZE_M = 8.0
RESOLUTION = 0.025
MAP_CELLS = int(MAP_SIZE_M / RESOLUTION)
ORIGIN = MAP_CELLS // 2

print("=" * 50)
print("FLOOR MAPPER WITH OBSTACLE AVOIDANCE")
print("=" * 50)

# Ultrasonic sensors will be initialized after Arduino connection
ultrasonic = None

# Motors
print("\n[1/4] Motors...")
arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2.5)
arduino.reset_input_buffer()

def motor(cmd):
    arduino.write(f"{cmd}\n".encode())
    arduino.flush()
    time.sleep(0.02)

motor("STOP")
print("  OK")

# Ultrasonic sensors (via Arduino)
if ULTRASONIC_AVAILABLE:
    print("\n[1.5/4] Ultrasonic sensors (via Arduino)...")
    try:
        ultrasonic = UltrasonicSensors(serial_conn=arduino)
    except Exception as e:
        print(f"  Ultrasonic not available: {e}")
        ultrasonic = None
else:
    print("\n[1.5/4] Ultrasonic module not available")

# LiDAR
print("\n[2/4] LiDAR...")
lidar = RPLidar('/dev/ttyUSB0')
lidar.stop()
lidar.stop_motor()
time.sleep(0.3)
lidar.start_motor()
time.sleep(1)
print("  OK")

# Navigation (encoder + IMU fusion)
nav = None
if NAVIGATION_AVAILABLE:
    print("\n[2.5/4] Navigation (encoders + IMU)...")
    try:
        nav = Navigation()
        # Pass arduino serial connection so encoders share it
        nav.init_sensors(calibrate_imu=True, serial_conn=arduino)
        nav.reset()
        print("  OK - Using encoder/IMU odometry")
    except Exception as e:
        print(f"  Navigation not available: {e}")
        nav = None
else:
    print("\n[2.5/4] Navigation not available - using timing estimates")

# Map
grid = np.full((MAP_CELLS, MAP_CELLS), 128, dtype=np.uint8)
robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
path = [(0, 0)]

def update_odometry():
    """Update robot position from navigation or return current values."""
    global robot_x, robot_y, robot_theta
    if nav:
        nav.update()
        pose = nav.get_pose()
        # Convert mm to meters
        robot_x = pose.x / 1000.0
        robot_y = pose.y / 1000.0
        robot_theta = pose.theta  # Already in radians

def w2g(x, y):
    gx = int(x / RESOLUTION) + ORIGIN
    gy = int(y / RESOLUTION) + ORIGIN
    return max(0, min(gx, MAP_CELLS-1)), max(0, min(gy, MAP_CELLS-1))

def update_map(scan):
    global grid
    rx, ry = w2g(robot_x, robot_y)
    grid[ry, rx] = 255

    for q, angle, dist in scan:
        if dist < 150 or dist > 5000:
            continue
        d = dist / 1000.0
        a = math.radians(angle) + robot_theta
        hx, hy = w2g(robot_x + d * math.cos(a), robot_y + d * math.sin(a))

        steps = max(abs(hx - rx), abs(hy - ry), 1)
        for i in range(steps):
            t = i / steps
            x = int(rx + t * (hx - rx))
            y = int(ry + t * (hy - ry))
            if 0 <= x < MAP_CELLS and 0 <= y < MAP_CELLS and grid[y, x] == 128:
                grid[y, x] = 255

        if 0 <= hx < MAP_CELLS and 0 <= hy < MAP_CELLS:
            grid[hy, hx] = 0

def get_distances(scan):
    """Get distances in front, left, right."""
    f, l, r = [], [], []
    for _, a, d in scan:
        if d <= 0:
            continue
        dm = d / 1000.0
        if a <= 25 or a >= 335:
            f.append(dm)
        elif 45 <= a <= 135:
            r.append(dm)
        elif 225 <= a <= 315:
            l.append(dm)

    front = min(f) if f else 9
    left = min(l) if l else 9
    right = min(r) if r else 9
    return front, left, right

def fwd(t):
    global robot_x, robot_y
    motor("FORWARD")
    time.sleep(t)
    motor("STOP")
    if nav:
        update_odometry()
    else:
        # Fallback: timing-based estimate
        d = t * 0.15
        robot_x += d * math.cos(robot_theta)
        robot_y += d * math.sin(robot_theta)
    path.append((robot_x, robot_y))

def turn_l():
    global robot_theta
    motor("LEFT")
    time.sleep(0.4)
    motor("STOP")
    if nav:
        update_odometry()
    else:
        robot_theta += 0.5

def turn_r():
    global robot_theta
    motor("RIGHT")
    time.sleep(0.4)
    motor("STOP")
    if nav:
        update_odometry()
    else:
        robot_theta -= 0.5

def save_map():
    os.makedirs("/home/bo/robot_pet/maps", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    img = np.zeros((MAP_CELLS, MAP_CELLS, 3), dtype=np.uint8)
    img[grid == 128] = (80, 80, 80)
    img[grid == 255] = (255, 255, 255)
    img[grid == 0] = (0, 0, 0)

    for px, py in path:
        gx, gy = w2g(px, py)
        cv2.circle(img, (gx, gy), 2, (0, 165, 255), -1)

    rx, ry = w2g(robot_x, robot_y)
    cv2.circle(img, (rx, ry), 5, (0, 255, 0), -1)
    img = cv2.flip(img, 0)

    p = f"/home/bo/robot_pet/maps/floor_{ts}.png"
    cv2.imwrite(p, img)
    return p

def get_ultrasonic_distances():
    """Get ultrasonic distances (cm). Returns (front_left, front_right, back) or None."""
    if ultrasonic is None:
        return None
    try:
        reading = ultrasonic.read_all()
        return reading.front_left, reading.front_right, reading.back
    except:
        return None

# Explore with obstacle avoidance
moves = 0
max_moves = 15
print(f"\n[3/4] Exploring ({max_moves} moves with obstacle avoidance)...")
scan_iter = lidar.iter_scans(max_buf_meas=8000, min_len=5)

try:
    while moves < max_moves:
        # Get scan
        try:
            scan = next(scan_iter)
        except:
            continue

        update_map(scan)
        front, left, right = get_distances(scan)

        free = np.sum(grid == 255)
        walls = np.sum(grid == 0)
        pct = (free + walls) / (MAP_CELLS * MAP_CELLS) * 100

        # Check ultrasonic sensors too
        us = get_ultrasonic_distances()
        us_str = ""
        if us:
            us_fl, us_fr, us_back = us
            us_str = f" US_FL:{us_fl:.0f} US_FR:{us_fr:.0f}"

        print(f"\r  Move {moves+1}/{max_moves} | {pct:.1f}% | F:{front:.2f} L:{left:.2f} R:{right:.2f}{us_str}  ", end="", flush=True)

        # OBSTACLE AVOIDANCE (LiDAR + Ultrasonic)
        # Check ultrasonic first (faster response, closer range)
        us_too_close = False
        if us:
            us_fl, us_fr, _ = us
            if (us_fl > 0 and us_fl < 25) or (us_fr > 0 and us_fr < 25):  # < 25cm
                us_too_close = True

        if us_too_close or front < 0.35:
            # Too close! Back up and turn
            motor("BACKWARD")
            time.sleep(0.2)
            motor("STOP")
            if nav:
                update_odometry()
            if left > right:
                turn_l()
            else:
                turn_r()
        elif front < 0.60:
            # Getting close, turn away
            if left > right:
                turn_l()
            else:
                turn_r()
        elif front < 1.0:
            # Medium distance - short move
            fwd(0.3)
        else:
            # Clear ahead, go forward
            fwd(0.5)

        moves += 1
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n  Stopped by user")
except Exception as e:
    print(f"\n  Error: {e}")

# Cleanup
print("\n\n[4/4] Saving map...")
motor("STOP")

try:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
except:
    pass

if ultrasonic:
    try:
        ultrasonic.close()
    except:
        pass

if nav:
    try:
        nav.close()
    except:
        pass

arduino.close()

map_path = save_map()

free = np.sum(grid == 255)
walls = np.sum(grid == 0)
pct = (free + walls) / (MAP_CELLS * MAP_CELLS) * 100

print(f"\n{'=' * 50}")
print(f"DONE!")
print(f"Map: {map_path}")
print(f"Explored: {pct:.1f}% | Walls: {walls}")
print("=" * 50)
