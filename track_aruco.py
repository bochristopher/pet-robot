#!/usr/bin/env python3
"""
ArUco marker tracking script.
Detects ArUco marker ID 0 (DICT_4X4_50) via OAK-D camera and publishes
Twist commands to /cmd_vel to rotate the robot toward the marker.
"""

import depthai as dai
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

MARKER_ID = 0
TURN_SPEED = 5.0  # constant angular speed (rad/s)
DEADZONE = 0.2  # normalized error threshold (0.0–1.0)
LOOP_HZ = 30


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
            q = device.getOutputQueue("mono", maxSize=1, blocking=False)
            loop_count = 0
            rate_t0 = time.monotonic()

            while rclpy.ok():
                t0 = time.monotonic()

                in_frame = q.tryGet()
                if in_frame is None:
                    time.sleep(0.005)
                    continue

                frame = in_frame.getCvFrame()
                gray = frame
                corners, ids, _ = detector.detectMarkers(gray)

                twist = Twist()

                if ids is not None:
                    for i, mid in enumerate(ids.flatten()):
                        if mid == MARKER_ID:
                            c = corners[i][0]
                            marker_cx = c[:, 0].mean()
                            marker_cy = c[:, 1].mean()
                            img_cx = frame.shape[1] / 2.0
                            img_half_w = frame.shape[1] / 2.0
                            error = (marker_cx - img_cx) / img_half_w
                            if abs(error) > DEADZONE:
                                twist.angular.z = -TURN_SPEED if error > 0 else TURN_SPEED
                            side = "LEFT" if marker_cx < img_cx else "RIGHT"
                            in_dz = abs(error) <= DEADZONE
                            node.get_logger().info(
                                f"Marker {MARKER_ID} at ({marker_cx:.1f}, {marker_cy:.1f}) | "
                                f"{side} of center (error={error:+.3f}){' [DEADZONE]' if in_dz else ''} | "
                                f"angular.z={twist.angular.z:+.2f} rad/s"
                            )
                            break

                pub.publish(twist)
                rclpy.spin_once(node, timeout_sec=0)

                loop_count += 1
                rate_elapsed = time.monotonic() - rate_t0
                if rate_elapsed >= 5.0:
                    node.get_logger().info(f"Loop rate: {loop_count / rate_elapsed:.1f} Hz (target {LOOP_HZ})")
                    loop_count = 0
                    rate_t0 = time.monotonic()

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
