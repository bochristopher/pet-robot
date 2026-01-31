#!/usr/bin/env python3
"""
ROS2 status reporter for Foxglove panels.
Publishes hearing/speaking/listening events for UI visibility.
"""

import threading
from typing import Optional


class RosReporter:
    """Optional ROS2 publisher for robot pet status."""

    def __init__(self):
        self.available = False
        self.node = None
        self._String = None
        self._Bool = None
        self._spin_thread: Optional[threading.Thread] = None

        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String, Bool
        except Exception as e:
            print(f"[ROS] ⚠️  ROS2 not available for status publishing: {e}")
            return

        try:
            rclpy.init(args=None)
            self.node = Node("robot_pet_status")
            self._String = String
            self._Bool = Bool

            self.pub_heard = self.node.create_publisher(String, "/ice_agent/heard", 10)
            self.pub_spoken = self.node.create_publisher(String, "/ice_agent/spoken", 10)
            self.pub_listening = self.node.create_publisher(Bool, "/ice_agent/listening", 10)

            self._spin_thread = threading.Thread(
                target=rclpy.spin, args=(self.node,), daemon=True
            )
            self._spin_thread.start()
            self.available = True
            print("[ROS] ✅ Status publishers ready (/ice_agent/heard, /ice_agent/spoken, /ice_agent/listening)")
        except Exception as e:
            print(f"[ROS] ⚠️  Failed to init ROS status publishers: {e}")

    def publish_heard(self, text: str):
        if not self.available:
            return
        msg = self._String()
        msg.data = text
        self.pub_heard.publish(msg)

    def publish_spoken(self, text: str):
        if not self.available:
            return
        msg = self._String()
        msg.data = text
        self.pub_spoken.publish(msg)

    def publish_listening(self, is_listening: bool):
        if not self.available:
            return
        msg = self._Bool()
        msg.data = bool(is_listening)
        self.pub_listening.publish(msg)

    def shutdown(self):
        if not self.available or not self.node:
            return
        try:
            import rclpy
            rclpy.shutdown()
        except Exception:
            pass

