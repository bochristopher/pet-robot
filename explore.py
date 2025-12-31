#!/usr/bin/env python3
"""
Simple Explorer with Obstacle Avoidance
Uses LiDAR + Ultrasonic sensors to navigate
"""

import time
import serial
import sys
import subprocess
import threading
from rplidar import RPLidar

# Try to import ultrasonic sensors
try:
    from ultrasonic import UltrasonicSensors
    ULTRASONIC_AVAILABLE = True
except:
    ULTRASONIC_AVAILABLE = False

print("=" * 50)
print("EXPLORER - Obstacle Avoidance Mode")
print("=" * 50)
sys.stdout.flush()

# Voice feedback (non-blocking)
def speak(text):
    """Speak text in background thread."""
    def _speak():
        try:
            subprocess.run(['espeak', '-s', '150', text],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()

speak("Explorer starting")

# Motors
print("\n[1/3] Motors...")
arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2.5)
arduino.reset_input_buffer()

def motor(cmd):
    arduino.write(f"{cmd}\n".encode())
    arduino.flush()
    # Wait for and consume response
    time.sleep(0.02)
    while arduino.in_waiting:
        arduino.readline()

motor("STOP")
print("  OK")

# Ultrasonic sensors
ultrasonic = None
if ULTRASONIC_AVAILABLE:
    print("\n[1.5/3] Ultrasonic sensors...")
    try:
        ultrasonic = UltrasonicSensors(serial_conn=arduino)
    except Exception as e:
        print(f"  Not available: {e}")

# LiDAR - proper initialization
print("\n[2/3] LiDAR...")
lidar = RPLidar('/dev/ttyUSB0')

# Full reset sequence
try:
    lidar.stop()
    lidar.stop_motor()
except:
    pass
time.sleep(0.5)

# Clear any stale data
lidar.clean_input()
lidar.start_motor()
time.sleep(2)  # Wait for motor to stabilize

# Get device info to verify connection
try:
    info = lidar.get_info()
    print(f"  Model: {info['model']}, Firmware: {info['firmware'][0]}.{info['firmware'][1]}")
except Exception as e:
    print(f"  Warning: Could not get info: {e}")

print("  OK")
sys.stdout.flush()

def get_distances(scan):
    """Get distances in front, left, right from LiDAR."""
    f, l, r = [], [], []
    for _, a, d in scan:
        if d <= 0:
            continue
        dm = d / 1000.0  # Convert to meters
        # Front: -30 to +30 degrees
        if a <= 30 or a >= 330:
            f.append(dm)
        # Right: 45-135 degrees
        elif 45 <= a <= 135:
            r.append(dm)
        # Left: 225-315 degrees
        elif 225 <= a <= 315:
            l.append(dm)

    front = min(f) if f else 9.0
    left = min(l) if l else 9.0
    right = min(r) if r else 9.0
    return front, left, right

def get_ultrasonic():
    """Get ultrasonic distances in cm."""
    if ultrasonic is None:
        return None, None, None
    try:
        reading = ultrasonic.read_all()
        return reading.front_left, reading.front_right, reading.back
    except:
        return None, None, None

def fwd(duration):
    motor("FORWARD")
    time.sleep(duration)
    motor("STOP")

def backup(duration, us_back=None):
    """Backup if safe, otherwise just turn."""
    # Check if rear is clear (us_back in cm, need > 30cm)
    if us_back is not None and us_back > 0 and us_back < 30:
        print(f" (rear blocked at {us_back}cm, turning instead)")
        return False  # Can't back up
    motor("BACKWARD")
    time.sleep(duration)
    motor("STOP")
    return True

def turn_left(duration=0.3):
    motor("LEFT")
    time.sleep(duration)
    motor("STOP")

def turn_right(duration=0.3):
    motor("RIGHT")
    time.sleep(duration)
    motor("STOP")

# Exploration loop
print("\n[3/3] Exploring (Ctrl+C to stop)...")
print("-" * 50)
sys.stdout.flush()
speak("Sensors ready. Starting exploration.")

moves = 0
stuck_count = 0
last_front = 9.0
scan_errors = 0

try:
    # Start scanning
    scan_iter = lidar.iter_scans(max_buf_meas=8000, min_len=5)

    while True:
        # Get LiDAR scan with error handling
        try:
            scan = next(scan_iter)
            scan_errors = 0  # Reset error count on success
        except StopIteration:
            print("\nLiDAR stopped, restarting...")
            lidar.stop()
            time.sleep(0.5)
            lidar.start_motor()
            time.sleep(1)
            scan_iter = lidar.iter_scans(max_buf_meas=8000, min_len=5)
            continue
        except Exception as e:
            scan_errors += 1
            if scan_errors > 10:
                print(f"\nToo many scan errors, resetting LiDAR...")
                lidar.stop()
                lidar.stop_motor()
                time.sleep(1)
                lidar.clean_input()
                lidar.start_motor()
                time.sleep(2)
                scan_iter = lidar.iter_scans(max_buf_meas=8000, min_len=5)
                scan_errors = 0
            continue

        front, left, right = get_distances(scan)
        us_fl, us_fr, us_back = get_ultrasonic()

        # Display status
        us_str = ""
        if us_fl is not None:
            us_str = f" | US: FL={us_fl:.0f} FR={us_fr:.0f}"
        print(f"\rMove {moves:3d} | F:{front:.2f}m L:{left:.2f}m R:{right:.2f}m{us_str}    ", end="")
        sys.stdout.flush()

        # Detect if stuck
        if abs(front - last_front) < 0.05 and front < 0.6:
            stuck_count += 1
        else:
            stuck_count = 0
        last_front = front

        # Check ultrasonic for very close obstacles
        us_too_close = False
        if us_fl is not None:
            if (us_fl > 0 and us_fl < 20) or (us_fr > 0 and us_fr < 20):
                us_too_close = True

        # STUCK RECOVERY
        if stuck_count > 5:
            print("\n  ** STUCK! Backing up and turning **")
            speak("I'm stuck")
            backup(0.6, us_back)
            # Turn toward more open side
            if left > right:
                turn_left(0.8)
            else:
                turn_right(0.8)
            stuck_count = 0
            moves += 1
            time.sleep(0.1)
            continue

        # OBSTACLE AVOIDANCE - always go toward farthest direction
        max_dist = max(front, left, right)

        if us_too_close or front < 0.25:
            # Too close! Try to backup (check rear first)
            print("\n  ** Too close!", end="")
            if moves % 10 == 0:  # Don't spam voice
                speak("obstacle")
            backed_up = backup(0.4, us_back)
            if not backed_up:
                # Rear blocked - just turn aggressively
                if left > right:
                    turn_left(0.6)
                else:
                    turn_right(0.6)
            else:
                # Backed up successfully, now turn
                if left > right:
                    turn_left(0.5)
                else:
                    turn_right(0.5)
        elif left > front and left > right and left > 0.5:
            # Left is farthest - turn left
            turn_left(0.3)
        elif right > front and right > left and right > 0.5:
            # Right is farthest - turn right
            turn_right(0.3)
        elif front < 0.40:
            # Front blocked, turn toward more open side
            if left > right:
                turn_left(0.35)
            else:
                turn_right(0.35)
        elif front < 0.70:
            # Medium distance - careful forward
            fwd(0.25)
        else:
            # Front is clear and farthest - go forward
            fwd(0.4)

        moves += 1
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\nStopped by user")
    speak("Stopping")

# Cleanup
print("\nCleaning up...")
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

arduino.close()

print(f"Done! Completed {moves} moves.")
print("=" * 50)
