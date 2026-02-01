#!/usr/bin/env python3
"""
ArUco marker tracking script.
Detects ArUco marker ID 0 (DICT_4X4_50) via OAK-D camera and publishes
Twist commands to /cmd_vel to rotate the robot toward the marker.
"""

import depthai as dai
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

MARKER_ID = 0
MARKER_SIZE = 0.05  # marker side length in meters
TURN_SPEED = 0.5  # constant angular speed (rad/s)
DEADZONE = 0.2  # normalized error threshold (0.0–1.0)
ALIGN_SPEED = 0.25  # lateral speed for alignment (m/s)
ALIGN_DEADZONE = 0.005  # lateral offset deadzone (meters)
TARGET_DIST = 0.1  # desired distance to marker (meters)
DIST_DEADZONE = 0.01  # distance deadzone (meters)
APPROACH_SPEED = 0.25  # forward/backward speed (m/s)
LOOP_HZ = 30

# 3D object points for a square marker centered at origin
_HALF = MARKER_SIZE / 2.0
OBJ_PTS = np.array([
    [-_HALF,  _HALF, 0],
    [ _HALF,  _HALF, 0],
    [ _HALF, -_HALF, 0],
    [-_HALF, -_HALF, 0],
], dtype=np.float32)


def _find_marker(corners, ids, marker_id=MARKER_ID):
    """Return the 4x2 corner array for *marker_id*, or None."""
    if ids is None:
        return None
    for i, mid in enumerate(ids.flatten()):
        if mid == marker_id:
            return corners[i][0]
    return None


def angular_track(corners, ids, frame_shape, camera_matrix, dist_coeffs):
    """Compute a twist velocity that rotates the robot toward the marker.

    Returns np.array([lx, ly, lz, ax, ay, az]).
    """
    vel = np.zeros(6)
    c = _find_marker(corners, ids)
    if c is None:
        return vel
    marker_cx = c[:, 0].mean()
    img_cx = frame_shape[1] / 2.0
    error = (marker_cx - img_cx) / img_cx
    if abs(error) > DEADZONE:
        vel[5] = -TURN_SPEED if error > 0 else TURN_SPEED
    return vel


def align(corners, ids, camera_matrix, dist_coeffs):
    """Compute a twist velocity that moves the robot laterally to stand
    directly in front of the marker (minimise the lateral offset from the
    marker's centre axis).

    Returns np.array([lx, ly, lz, ax, ay, az]).
    """
    vel = np.zeros(6)
    c = _find_marker(corners, ids)
    if c is None:
        return vel
    ok, rvec, tvec = cv2.solvePnP(OBJ_PTS, c, camera_matrix, dist_coeffs)
    if not ok:
        return vel
    lateral_offset = float(tvec[0])
    if abs(lateral_offset) > ALIGN_DEADZONE:
        vel[1] = ALIGN_SPEED if lateral_offset > 0 else -ALIGN_SPEED
    return vel


def keep_distance(corners, ids, camera_matrix, dist_coeffs, target_dist=TARGET_DIST):
    """Move forward/backward to maintain target_dist from the marker.

    Returns np.array([lx, ly, lz, ax, ay, az]).
    """
    vel = np.zeros(6)
    c = _find_marker(corners, ids)
    if c is None:
        return vel
    ok, rvec, tvec = cv2.solvePnP(OBJ_PTS, c, camera_matrix, dist_coeffs)
    if not ok:
        return vel
    dist = float(tvec[2])  # Z axis = distance from camera
    error = dist - target_dist
    if abs(error) > DIST_DEADZONE:
        vel[0] = APPROACH_SPEED if error > 0 else -APPROACH_SPEED
    return vel


def _vel_to_twist(vel):
    """Convert a 6-element numpy velocity to a Twist message."""
    twist = Twist()
    twist.linear.x = float(vel[0])
    twist.linear.y = float(vel[1])
    twist.linear.z = float(vel[2])
    twist.angular.x = float(vel[3])
    twist.angular.y = float(vel[4])
    twist.angular.z = float(vel[5])
    return twist


def create_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("mono")
    cam.out.link(xout.input)
    return pipeline


def main():
    rclpy.init()
    node = Node("aruco_tracker")
    pub = node.create_publisher(Twist, "cmd_vel", 10)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    pipeline = create_pipeline()
    period = 1.0 / LOOP_HZ

    node.get_logger().info(
        f"Tracking ArUco ID {MARKER_ID} | speed={TURN_SPEED} | deadzone={DEADZONE} | loop={LOOP_HZ} Hz"
    )

    try:
        with dai.Device(pipeline) as device:
            calib = device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 640, 480)
            camera_matrix = np.array(intrinsics, dtype=np.float32)
            dist_coeffs = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B), dtype=np.float32)

            q = device.getOutputQueue("mono", maxSize=1, blocking=False)

            stage = 1
            stage2_start = None
            STAGE1_DIST = 0.3
            STAGE2_FROM = 0.3
            STAGE2_TO = 0.1
            STAGE2_DURATION = 5.0

            while rclpy.ok():
                t0 = time.monotonic()

                in_frame = q.tryGet()
                if in_frame is None:
                    time.sleep(0.005)
                    continue

                frame = in_frame.getCvFrame()
                gray = frame
                corners, ids, _ = detector.detectMarkers(gray)

                if stage == 1:
                    vel = angular_track(corners, ids, frame.shape, camera_matrix, dist_coeffs)
                    vel += align(corners, ids, camera_matrix, dist_coeffs)
                    vel += keep_distance(corners, ids, camera_matrix, dist_coeffs, target_dist=STAGE1_DIST)
                    twist = _vel_to_twist(vel)
                    pub.publish(twist)

                    # Check if centered and at distance
                    centered = abs(vel[5]) == 0  # no angular correction
                    at_dist = abs(vel[0]) == 0    # no forward/backward correction
                    if centered and at_dist and _find_marker(corners, ids) is not None:
                        print("Stage 1 complete: centered and 0.3m away. Starting stage 2.")
                        stage = 2
                        stage2_start = time.monotonic()

                elif stage == 2:
                    elapsed_s2 = time.monotonic() - stage2_start
                    t_frac = min(elapsed_s2 / STAGE2_DURATION, 1.0)
                    target = STAGE2_FROM + (STAGE2_TO - STAGE2_FROM) * t_frac

                    vel = angular_track(corners, ids, frame.shape, camera_matrix, dist_coeffs)
                    vel += align(corners, ids, camera_matrix, dist_coeffs)
                    vel += keep_distance(corners, ids, camera_matrix, dist_coeffs, target_dist=target)
                    twist = _vel_to_twist(vel)
                    pub.publish(twist)

                    if t_frac >= 1.0:
                        print("Stage 2 complete. Done.")
                        break

                rclpy.spin_once(node, timeout_sec=0)

                elapsed = time.monotonic() - t0
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot
        pub.publish(Twist())
        rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
