#!/usr/bin/env python3
"""
Data recorder for robot_pet.
Records cmd_vel velocities (with timestamps) to a text file and
camera frames (with timestamps) to an images subfolder, all at 30 Hz.

Keyboard controls:
    W/S  - linear.x  (forward / backward)
    A/D  - linear.y  (left / right)
    Q/E  - angular.z (turn left / turn right)
    Space - stop (zero all velocities)
    K    - exit recording

Usage:
    python3 recorder.py /path/to/output_folder
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

import cv2
import depthai as dai
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

LOOP_HZ = 30
LINEAR_SPEED = 0.3   # m/s per key
ANGULAR_SPEED = 1.0  # rad/s per key


def get_key(timeout=0.0):
    """Non-blocking single-character read from stdin."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


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
    parser = argparse.ArgumentParser(description="Record robot data at 30 Hz")
    parser.add_argument("folder", help="Output folder for recorded data")
    args = parser.parse_args()

    # Create output directories
    images_dir = os.path.join(args.folder, "images")
    os.makedirs(images_dir, exist_ok=True)

    vel_path = os.path.join(args.folder, "velocities.txt")

    # ROS2 setup — publish and subscribe cmd_vel
    rclpy.init()
    node = Node("recorder")

    latest_twist = Twist()
    pub = node.create_publisher(Twist, "cmd_vel", 10)

    def twist_cb(msg):
        nonlocal latest_twist
        latest_twist = msg

    node.create_subscription(Twist, "cmd_vel", twist_cb, 10)

    pipeline = create_pipeline()
    period = 1.0 / LOOP_HZ

    node.get_logger().info(
        f"Recording to {args.folder} at {LOOP_HZ} Hz  (K or Ctrl-C to stop)"
    )
    node.get_logger().info(
        "Keys: W/S=fwd/back  A/D=left/right  Q/E=turn  Space=stop  K=exit"
    )

    frame_count = 0

    # Put terminal in raw mode for non-blocking key reads
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("mono", maxSize=1, blocking=False)

            with open(vel_path, "w") as vel_file:
                vel_file.write(
                    "# timestamp linear_x linear_y linear_z "
                    "angular_x angular_y angular_z\n"
                )

                cmd_twist = Twist()

                while rclpy.ok():
                    t0 = time.monotonic()
                    ts = time.time()

                    # Handle keyboard input
                    cmd_twist = Twist()
                    key = get_key()
                    if key is not None:
                        k = key.lower()
                        print(f"[key] '{key}'")
                        if k == "k":
                            break
                        elif k == "w":
                            cmd_twist.linear.x = LINEAR_SPEED
                        elif k == "s":
                            cmd_twist.linear.x = -LINEAR_SPEED
                        elif k == "a":
                            cmd_twist.linear.y = LINEAR_SPEED
                        elif k == "d":
                            cmd_twist.linear.y = -LINEAR_SPEED
                        elif k == "q":
                            cmd_twist.angular.z = ANGULAR_SPEED
                        elif k == "e":
                            cmd_twist.angular.z = -ANGULAR_SPEED
                    pub.publish(cmd_twist)

                    # Drain latest ROS messages
                    rclpy.spin_once(node, timeout_sec=0)

                    # Log velocity
                    tw = latest_twist
                    vel_file.write(
                        f"{ts:.6f} "
                        f"{tw.linear.x:.6f} {tw.linear.y:.6f} {tw.linear.z:.6f} "
                        f"{tw.angular.x:.6f} {tw.angular.y:.6f} {tw.angular.z:.6f}\n"
                    )

                    # Grab and save camera frame
                    in_frame = q.tryGet()
                    if in_frame is not None:
                        frame = in_frame.getCvFrame()
                        img_name = f"{ts:.6f}.png"
                        cv2.imwrite(os.path.join(images_dir, img_name), frame)
                        frame_count += 1

                    elapsed = time.monotonic() - t0
                    sleep_time = period - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        # Stop the robot
        pub.publish(Twist())
        rclpy.spin_once(node, timeout_sec=0.05)
        node.get_logger().info(
            f"Stopped. Wrote {frame_count} images to {images_dir}"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
