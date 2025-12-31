#!/usr/bin/env python3
"""
Smart Explorer with Path Planning
- Position tracking (encoders + IMU)
- Occupancy grid memory
- Frontier-based exploration
- Goal-directed navigation
"""

import time
import serial
import sys
import subprocess
import threading
import math
import os
import numpy as np
from collections import deque
from rplidar import RPLidar

sys.path.insert(0, '/home/bo/robot_pet')

# Optional imports - using new package structure
try:
    from sensors.ultrasonic import UltrasonicSensors
    ULTRASONIC_AVAILABLE = True
except:
    ULTRASONIC_AVAILABLE = False

try:
    from sensors.imu_mpu6050 import MPU6050
    IMU_AVAILABLE = True
except:
    IMU_AVAILABLE = False

try:
    from voice.elevenlabs_speaker import speak as elevenlabs_speak
    ELEVENLABS_AVAILABLE = True
except:
    ELEVENLABS_AVAILABLE = False

try:
    from sensors.camera_obstacle import CameraObstacleDetector
    CAMERA_AVAILABLE = True
except:
    CAMERA_AVAILABLE = False

try:
    from perception.face_recognition_simple import FaceRecognizer
    FACE_RECOGNITION_AVAILABLE = True
except:
    FACE_RECOGNITION_AVAILABLE = False

print("=" * 50)
print("SMART EXPLORER - Path Planning Mode")
print("=" * 50)

# ============ CONFIGURATION ============
GRID_SIZE = 200          # 200x200 grid
GRID_RESOLUTION = 0.05   # 5cm per cell
ROBOT_RADIUS = 0.15      # 15cm robot radius
GOAL_REACHED_DIST = 0.2  # 20cm to consider goal reached
FRONTIER_MIN_SIZE = 3    # Minimum frontier size to consider

# Grid center is robot start position
GRID_CENTER = GRID_SIZE // 2

# ============ VOICE WITH PERSONALITY ============
import random

last_speak_time = 0
last_phrase_key = None

# Personality phrases - sassy robot pet!
PHRASES = {
    'startup': [
        "Alright, let's do this!",
        "Adventure mode activated!",
        "Watch and learn, humans.",
        "Time to show off my moves!",
        "Miss me? I'm back!",
    ],
    'obstacle': [
        "Excuse me, who put that there?",
        "Rude! Something's in my way.",
        "Nope, not today!",
        "Oh please, I'll just go around.",
        "Nice try, obstacle!",
        "You think that can stop me?",
    ],
    'stuck': [
        "Okay, this is embarrassing.",
        "Don't look at me right now!",
        "Ugh, seriously?",
        "This is fine. Everything is fine.",
        "A minor setback!",
        "We don't talk about this.",
    ],
    'unstuck': [
        "Ha! Never doubted myself.",
        "And I'm back, baby!",
        "That was on purpose, obviously.",
        "Nailed it!",
        "Too easy!",
    ],
    'new_goal': [
        "Ooh, what do we have here?",
        "Something interesting this way!",
        "My sensors are tingling!",
        "Let me investigate!",
        "New territory? Don't mind if I do!",
    ],
    'reached': [
        "Another conquest!",
        "Been there, done that!",
        "Check that off the list!",
        "I'm basically a genius.",
        "Too easy!",
    ],
    'all_explored': [
        "I own this place now.",
        "Conquered! What's next?",
        "Is that all you've got?",
        "This place has no secrets from me!",
    ],
    'stopping': [
        "Fine, I'll take a break. You're welcome!",
        "Until next time, peasants!",
        "I need my beauty rest anyway.",
        "Peace out!",
        "Don't miss me too much!",
    ],
    'camera_obstacle': [
        "Hold up, I see something!",
        "My eyes don't lie!",
        "Suspicious object detected!",
        "What's that doing here?",
    ],
    'greet_bo': [
        "Hey Bo!",
        "Hi Bo, what's up?",
        "Oh hi Bo!",
        "Bo! Good to see you!",
        "There's my favorite human!",
    ],
    'greet_stranger': [
        "Who are you?",
        "I don't recognize you.",
        "New face detected!",
        "Hello stranger!",
    ],
}

def speak(key_or_text, min_interval=2.0):
    """Speak with personality. Pass a key from PHRASES or custom text."""
    global last_speak_time, last_phrase_key
    now = time.time()
    if now - last_speak_time < min_interval:
        return

    # Get phrase - either from PHRASES dict or use as-is
    if key_or_text in PHRASES:
        phrases = PHRASES[key_or_text]
        text = random.choice(phrases)
        # Avoid repeating the exact same phrase
        if len(phrases) > 1 and last_phrase_key == key_or_text:
            while text == getattr(speak, '_last_text', None):
                text = random.choice(phrases)
        speak._last_text = text
        last_phrase_key = key_or_text
    else:
        text = key_or_text

    last_speak_time = now

    def _speak():
        try:
            if ELEVENLABS_AVAILABLE:
                elevenlabs_speak(text, blocking=False)
            else:
                env = os.environ.copy()
                env['ALSA_CARD'] = '0'
                subprocess.run(['espeak', '-s', '150', text],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env=env)
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()

speak("startup")

# ============ OCCUPANCY GRID ============
# 0 = unknown, 1 = free, 2 = obstacle
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

def world_to_grid(x, y):
    """Convert world coords (meters) to grid coords."""
    gx = int(GRID_CENTER + x / GRID_RESOLUTION)
    gy = int(GRID_CENTER + y / GRID_RESOLUTION)
    return max(0, min(GRID_SIZE-1, gx)), max(0, min(GRID_SIZE-1, gy))

def grid_to_world(gx, gy):
    """Convert grid coords to world coords (meters)."""
    x = (gx - GRID_CENTER) * GRID_RESOLUTION
    y = (gy - GRID_CENTER) * GRID_RESOLUTION
    return x, y

def mark_free(x, y):
    """Mark a cell as free."""
    gx, gy = world_to_grid(x, y)
    if grid[gx, gy] == 0:
        grid[gx, gy] = 1

def mark_obstacle(x, y):
    """Mark a cell as obstacle."""
    gx, gy = world_to_grid(x, y)
    grid[gx, gy] = 2

def update_grid_from_scan(robot_x, robot_y, robot_theta, scan):
    """Update occupancy grid from LiDAR scan using ray tracing."""
    for _, angle_deg, dist_mm in scan:
        if dist_mm <= 0 or dist_mm > 5000:
            continue

        dist = dist_mm / 1000.0
        angle = math.radians(angle_deg) + robot_theta

        # Endpoint of ray
        end_x = robot_x + dist * math.cos(angle)
        end_y = robot_y + dist * math.sin(angle)

        # Ray trace - mark cells along ray as free
        steps = int(dist / GRID_RESOLUTION)
        for i in range(steps):
            t = i / max(steps, 1)
            px = robot_x + t * (end_x - robot_x)
            py = robot_y + t * (end_y - robot_y)
            mark_free(px, py)

        # Mark endpoint as obstacle (if within range)
        if dist < 4.0:
            mark_obstacle(end_x, end_y)

def find_frontiers(robot_x, robot_y):
    """Find frontier cells (free cells next to unknown)."""
    frontiers = []
    gx_robot, gy_robot = world_to_grid(robot_x, robot_y)

    for gx in range(1, GRID_SIZE-1):
        for gy in range(1, GRID_SIZE-1):
            if grid[gx, gy] != 1:  # Not free
                continue

            # Check if adjacent to unknown
            has_unknown = False
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                if grid[gx+dx, gy+dy] == 0:
                    has_unknown = True
                    break

            if has_unknown:
                # Calculate distance from robot
                wx, wy = grid_to_world(gx, gy)
                dist = math.sqrt((wx - robot_x)**2 + (wy - robot_y)**2)
                if dist > 0.3:  # Not too close
                    frontiers.append((wx, wy, dist))

    return frontiers

def find_best_goal(robot_x, robot_y, robot_theta):
    """Find the best exploration goal."""
    frontiers = find_frontiers(robot_x, robot_y)

    if not frontiers:
        return None

    # Score frontiers by distance and direction
    best_score = -float('inf')
    best_goal = None

    for fx, fy, dist in frontiers:
        # Prefer medium distance (not too close, not too far)
        dist_score = -abs(dist - 1.5)  # Optimal distance ~1.5m

        # Prefer direction we're facing
        angle_to_frontier = math.atan2(fy - robot_y, fx - robot_x)
        angle_diff = abs(angle_to_frontier - robot_theta)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        direction_score = -angle_diff

        score = dist_score * 0.5 + direction_score * 0.5

        if score > best_score:
            best_score = score
            best_goal = (fx, fy)

    return best_goal

def is_path_clear(robot_x, robot_y, goal_x, goal_y):
    """Check if path to goal is clear of obstacles."""
    dist = math.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
    steps = int(dist / GRID_RESOLUTION) + 1

    for i in range(steps):
        t = i / max(steps, 1)
        px = robot_x + t * (goal_x - robot_x)
        py = robot_y + t * (goal_y - robot_y)
        gx, gy = world_to_grid(px, py)

        # Check cell and neighbors for obstacles
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                ngx, ngy = gx + dx, gy + dy
                if 0 <= ngx < GRID_SIZE and 0 <= ngy < GRID_SIZE:
                    if grid[ngx, ngy] == 2:
                        return False
    return True

# ============ HARDWARE SETUP ============
print("\n[1/4] Arduino...")
arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2.5)
arduino.reset_input_buffer()

def motor(cmd):
    arduino.write(f"{cmd}\n".encode())
    arduino.flush()
    time.sleep(0.02)
    while arduino.in_waiting:
        arduino.readline()

def get_encoders():
    """Get encoder counts."""
    arduino.write(b"ENCODERS\n")
    arduino.flush()
    time.sleep(0.02)
    try:
        while arduino.in_waiting:
            resp = arduino.readline().decode().strip()
            if resp.startswith("ENC:"):
                parts = resp[4:].split(",")
                return int(parts[0]), int(parts[1])
    except:
        pass
    return 0, 0

motor("STOP")
motor("RESET")  # Reset encoders
print("  OK")

# Ultrasonic
ultrasonic = None
if ULTRASONIC_AVAILABLE:
    print("\n[2/4] Ultrasonic...")
    try:
        ultrasonic = UltrasonicSensors(serial_conn=arduino)
    except Exception as e:
        print(f"  Not available: {e}")

# IMU
imu = None
if IMU_AVAILABLE:
    print("\n[2.5/4] IMU...")
    try:
        imu = MPU6050(bus_num=7)
        imu.calibrate(50)
        imu.reset_yaw()
        print("  OK")
    except Exception as e:
        print(f"  Not available: {e}")

# Camera (local obstacle detection)
camera = None
if CAMERA_AVAILABLE:
    print("\n[2.7/4] Camera...")
    try:
        camera = CameraObstacleDetector()
        if camera.open():
            print("  OK")
        else:
            print("  Failed to open camera")
            camera = None
    except Exception as e:
        print(f"  Not available: {e}")

# Face Recognition
face_recognizer = None
last_greeted = {}  # Track when we greeted each person
if FACE_RECOGNITION_AVAILABLE:
    print("\n[2.8/4] Face Recognition...")
    try:
        face_recognizer = FaceRecognizer()
        if face_recognizer.trained:
            print(f"  OK - knows {len(face_recognizer.labels)} people")
        else:
            print("  No faces enrolled yet")
            face_recognizer = None
    except Exception as e:
        print(f"  Not available: {e}")

# LiDAR
print("\n[3/4] LiDAR...")
lidar = RPLidar('/dev/ttyUSB0')
try:
    lidar.stop()
    lidar.stop_motor()
except:
    pass
time.sleep(0.5)
lidar.clean_input()
lidar.start_motor()
time.sleep(2)
print("  OK")

# ============ ODOMETRY ============
robot_x = 0.0
robot_y = 0.0
robot_theta = 0.0
last_left_enc = 0
last_right_enc = 0

WHEEL_BASE = 0.20       # 20cm between wheels
TICKS_PER_METER = 1000  # Encoder ticks per meter (adjust!)

# Stuck detection
last_move_direction = None  # "FORWARD", "LEFT", "RIGHT", "BACKWARD"
stuck_counter = 0

def update_odometry():
    """Update robot position from encoders and IMU."""
    global robot_x, robot_y, robot_theta
    global last_left_enc, last_right_enc

    left_enc, right_enc = get_encoders()

    # Calculate deltas
    d_left = (left_enc - last_left_enc) / TICKS_PER_METER
    d_right = (right_enc - last_right_enc) / TICKS_PER_METER

    last_left_enc = left_enc
    last_right_enc = right_enc

    # Get heading from IMU if available
    if imu:
        try:
            robot_theta = math.radians(imu.get_heading())
        except:
            # Fall back to encoder-based heading
            d_theta = (d_right - d_left) / WHEEL_BASE
            robot_theta += d_theta
    else:
        d_theta = (d_right - d_left) / WHEEL_BASE
        robot_theta += d_theta

    # Update position
    d_center = (d_left + d_right) / 2
    robot_x += d_center * math.cos(robot_theta)
    robot_y += d_center * math.sin(robot_theta)

# ============ MOVEMENT ============
def get_ultrasonic():
    if ultrasonic is None:
        return None, None, None
    try:
        r = ultrasonic.read_all()
        return r.front_left, r.front_right, r.back
    except:
        return None, None, None

def forward(duration):
    global stuck_counter, last_move_direction
    before_x, before_y = robot_x, robot_y

    motor("FORWARD")
    time.sleep(duration)
    motor("STOP")
    update_odometry()

    # Check if we actually moved
    dist_moved = math.sqrt((robot_x - before_x)**2 + (robot_y - before_y)**2)
    if dist_moved < 0.02:  # Less than 2cm
        stuck_counter += 1
        last_move_direction = "FORWARD"
    else:
        stuck_counter = 0

def backward(duration):
    global stuck_counter, last_move_direction
    before_x, before_y = robot_x, robot_y

    motor("BACKWARD")
    time.sleep(duration)
    motor("STOP")
    update_odometry()

    dist_moved = math.sqrt((robot_x - before_x)**2 + (robot_y - before_y)**2)
    if dist_moved < 0.02:
        stuck_counter += 1
        last_move_direction = "BACKWARD"
    else:
        stuck_counter = 0

def turn_left(duration):
    global stuck_counter, last_move_direction
    before_theta = robot_theta

    motor("LEFT")
    time.sleep(duration)
    motor("STOP")
    update_odometry()

    # Check if heading changed
    theta_diff = abs(robot_theta - before_theta)
    if theta_diff < 0.05:  # Less than ~3 degrees
        stuck_counter += 1
        last_move_direction = "LEFT"
    else:
        stuck_counter = 0

def turn_right(duration):
    global stuck_counter, last_move_direction
    before_theta = robot_theta

    motor("RIGHT")
    time.sleep(duration)
    motor("STOP")
    update_odometry()

    theta_diff = abs(robot_theta - before_theta)
    if theta_diff < 0.05:
        stuck_counter += 1
        last_move_direction = "RIGHT"
    else:
        stuck_counter = 0

def turn_to_angle(target_theta):
    """Turn to face a specific angle."""
    diff = target_theta - robot_theta
    # Normalize to -pi to pi
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi

    if abs(diff) < 0.15:  # ~8 degrees tolerance
        return

    turn_time = min(abs(diff) * 0.3, 0.5)  # Proportional turn
    if diff > 0:
        turn_left(turn_time)
    else:
        turn_right(turn_time)

def navigate_to_goal(goal_x, goal_y, dist):
    """Navigate toward goal with obstacle avoidance."""
    global robot_theta

    # Calculate angle to goal
    angle_to_goal = math.atan2(goal_y - robot_y, goal_x - robot_x)
    angle_diff = angle_to_goal - robot_theta

    # Normalize
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    # Turn toward goal if needed
    if abs(angle_diff) > 0.3:  # >17 degrees
        if angle_diff > 0:
            turn_left(0.25)
        else:
            turn_right(0.25)
        return "TURNING"

    # Move forward
    move_time = min(dist * 0.8, 0.8)  # Longer moves toward goals
    forward(move_time)
    return "FORWARD"

# ============ MAIN LOOP ============
print("\n[4/4] Exploring with path planning...")
print("-" * 50)
speak("startup")

scan_iter = lidar.iter_scans(max_buf_meas=8000, min_len=5)
moves = 0
current_goal = None
goal_attempts = 0
last_action = "START"
exploration_complete = False

try:
    while True:
        # Get LiDAR scan
        try:
            scan = next(scan_iter)
        except StopIteration:
            lidar.stop()
            time.sleep(0.5)
            lidar.start_motor()
            time.sleep(1)
            scan_iter = lidar.iter_scans(max_buf_meas=8000, min_len=5)
            continue
        except:
            continue

        # Update odometry
        update_odometry()

        # Update map
        update_grid_from_scan(robot_x, robot_y, robot_theta, scan)

        # Get immediate distances for safety
        sectors = {'front': [], 'left': [], 'right': []}
        for _, angle, dist in scan:
            if dist <= 0:
                continue
            dm = dist / 1000.0
            a = angle % 360
            if a < 30 or a >= 330:
                sectors['front'].append(dm)
            elif 225 <= a <= 315:
                sectors['left'].append(dm)
            elif 45 <= a <= 135:
                sectors['right'].append(dm)

        front = min(sectors['front']) if sectors['front'] else 9.0
        left = min(sectors['left']) if sectors['left'] else 9.0
        right = min(sectors['right']) if sectors['right'] else 9.0

        us_fl, us_fr, us_back = get_ultrasonic()

        # Status display
        goal_str = ""
        if current_goal:
            gx, gy = current_goal
            goal_dist = math.sqrt((gx - robot_x)**2 + (gy - robot_y)**2)
            goal_str = f" Goal:{goal_dist:.1f}m"

        explored = np.sum(grid > 0)
        print(f"\rMove {moves:3d} | Pos:({robot_x:.1f},{robot_y:.1f}) θ:{math.degrees(robot_theta):.0f}° | F:{front:.1f} L:{left:.1f} R:{right:.1f}{goal_str} | Mapped:{explored} [{last_action}]     ", end="")
        sys.stdout.flush()

        # Safety checks
        us_front_close = us_fl is not None and ((us_fl > 0 and us_fl < 20) or (us_fr > 0 and us_fr < 20))
        us_back_close = us_back is not None and us_back > 0 and us_back < 25

        # CAMERA CHECK - real-time obstacle detection (every 3 moves for responsiveness)
        camera_blocked = False
        if camera is not None and moves % 3 == 0:
            try:
                obs = camera.detect_obstacles()
                if obs:
                    dist = obs.get('obstacle_distance', 'clear')
                    if dist in ['close', 'medium'] or not obs.get('path_clear', True):
                        camera_blocked = True
                        action = camera.get_recommended_action(obs)
                        if dist == 'close':
                            print(f"\n  ** CAMERA: obstacle {dist}! -> {action} **")
                            speak("camera_obstacle", min_interval=3)
            except Exception as e:
                pass  # Camera check failed, continue with LiDAR

        # FACE RECOGNITION - greet people we see (every 10 moves)
        if face_recognizer is not None and moves % 10 == 0:
            try:
                name, certainty = face_recognizer.who_is_there()
                if name and certainty > 20:
                    now = time.time()
                    last_time = last_greeted.get(name, 0)
                    if now - last_time > 60:  # Don't greet same person within 60s
                        last_greeted[name] = now
                        if name.lower() == "bo":
                            speak("greet_bo", min_interval=5)
                            print(f"\n  ** Recognized: {name} ({certainty:.0f}%)! **")
                        elif name == "stranger":
                            speak("greet_stranger", min_interval=10)
                            print(f"\n  ** Unknown person detected! **")
                        else:
                            speak(f"Hi {name}!", min_interval=5)
                            print(f"\n  ** Recognized: {name} ({certainty:.0f}%)! **")
            except Exception as e:
                pass  # Face check failed, continue

        # EMERGENCY OBSTACLE AVOIDANCE
        if us_front_close or front < 0.25 or camera_blocked:
            if not us_back_close:
                backward(0.3)
            if left > right:
                turn_left(0.4)
            else:
                turn_right(0.4)
            current_goal = None  # Cancel current goal
            last_action = "AVOID"
            speak("obstacle", min_interval=3)
            moves += 1
            continue

        # STUCK RECOVERY - wheels spinning but robot not moving
        if stuck_counter >= 3:
            speak("stuck", min_interval=5)
            print(f"\n  ** STUCK ({last_move_direction})! Recovering... **")

            if last_move_direction == "FORWARD":
                if not us_back_close:
                    backward(0.5)
                if left > right:
                    turn_left(0.6)
                else:
                    turn_right(0.6)
            elif last_move_direction == "BACKWARD":
                forward(0.3)
                turn_right(0.5)
            elif last_move_direction == "LEFT":
                turn_right(0.6)
            elif last_move_direction == "RIGHT":
                turn_left(0.6)
            else:
                if not us_back_close:
                    backward(0.4)
                turn_right(0.7)

            stuck_counter = 0
            current_goal = None  # Find new goal
            last_action = "UNSTUCK"
            speak("unstuck", min_interval=2)
            moves += 1
            continue

        # GOAL-BASED NAVIGATION
        if current_goal is None or goal_attempts > 20:
            # Find new goal
            current_goal = find_best_goal(robot_x, robot_y, robot_theta)
            goal_attempts = 0

            if current_goal is None:
                # No frontiers - exploration might be complete
                if not exploration_complete:
                    speak("all_explored", min_interval=10)
                    exploration_complete = True
                # Wander randomly
                if front > 0.8:
                    forward(0.5)
                    last_action = "WANDER"
                elif left > right:
                    turn_left(0.4)
                    last_action = "WANDER-L"
                else:
                    turn_right(0.4)
                    last_action = "WANDER-R"
                moves += 1
                continue
            else:
                exploration_complete = False
                speak("new_goal", min_interval=5)

        # Navigate to goal
        gx, gy = current_goal
        goal_dist = math.sqrt((gx - robot_x)**2 + (gy - robot_y)**2)

        if goal_dist < GOAL_REACHED_DIST:
            # Reached goal
            speak("reached", min_interval=3)
            current_goal = None
            last_action = "REACHED"
        elif not is_path_clear(robot_x, robot_y, gx, gy):
            # Path blocked - try to go around
            if front > 0.6:
                forward(0.5)
                last_action = "CLEAR-FWD"
            elif left > right:
                turn_left(0.4)
                last_action = "CLEAR-L"
            else:
                turn_right(0.4)
                last_action = "CLEAR-R"
            goal_attempts += 1
            if goal_attempts > 10:
                current_goal = None  # Give up on this goal
        elif front < 0.5:
            # Immediate obstacle - go around
            if left > right:
                turn_left(0.35)
                last_action = "DETOUR-L"
            else:
                turn_right(0.35)
                last_action = "DETOUR-R"
            goal_attempts += 1
        else:
            # Navigate toward goal
            last_action = navigate_to_goal(gx, gy, goal_dist)
            goal_attempts += 1

        moves += 1
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\nStopped by user")
    speak("stopping")

# Save map
print("\nSaving map...")
try:
    np.save('/home/bo/robot_pet/data/exploration_map.npy', grid)
    print(f"  Saved to data/exploration_map.npy ({np.sum(grid > 0)} cells explored)")
except Exception as e:
    print(f"  Could not save: {e}")

# Cleanup
print("Cleaning up...")
motor("STOP")

try:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
except:
    pass

if camera:
    try:
        camera.close()
    except:
        pass

if face_recognizer:
    try:
        face_recognizer.close_camera()
    except:
        pass

arduino.close()

print(f"Done! Completed {moves} moves.")
print(f"Final position: ({robot_x:.2f}, {robot_y:.2f}) heading {math.degrees(robot_theta):.1f}°")
print("=" * 50)
