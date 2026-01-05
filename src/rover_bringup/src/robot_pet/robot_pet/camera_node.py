#!/usr/bin/env python3
"""
Camera Node with AI Depth Estimation for obstacle detection.
Uses MiDaS for monocular depth estimation from regular webcam.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time

# Try to import torch for depth estimation
DEPTH_AVAILABLE = False
try:
    import torch
    DEPTH_AVAILABLE = True
except ImportError:
    pass


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Parameters
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15)
        self.declare_parameter('enable_depth', False)  # AI depth estimation
        self.declare_parameter('obstacle_threshold', 0.3)  # Depth threshold for obstacles

        device = self.get_parameter('device').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        self.enable_depth = self.get_parameter('enable_depth').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value

        # OpenCV camera
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera: {device}')
            return

        self.get_logger().info(f'Camera opened: {device} ({width}x{height} @ {fps}fps)')

        # CV Bridge
        self.bridge = CvBridge()

        # Publishers
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, 'camera/depth', 10)
        self.obstacles_pub = self.create_publisher(Float32MultiArray, 'camera/obstacles', 10)

        # Load depth model if enabled
        self.depth_model = None
        self.depth_transform = None
        if self.enable_depth and DEPTH_AVAILABLE:
            self.load_depth_model()

        # Timer for frame capture
        self.timer = self.create_timer(1.0 / fps, self.capture_frame)

        self.get_logger().info('Camera Node ready!')

    def load_depth_model(self):
        """Load MiDaS depth estimation model."""
        try:
            self.get_logger().info('Loading MiDaS depth model (this may take a moment)...')

            # Use small model for speed
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.depth_model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self.depth_model = self.depth_model.cuda()
                self.get_logger().info('Using CUDA for depth estimation')

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.depth_transform = midas_transforms.small_transform

            self.get_logger().info('MiDaS depth model loaded!')
        except Exception as e:
            self.get_logger().error(f'Failed to load depth model: {e}')
            self.depth_model = None

    def capture_frame(self):
        """Capture and process a frame."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Publish raw image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera_link'
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')

        # Depth estimation if enabled
        if self.depth_model is not None:
            self.process_depth(frame)
        else:
            # Simple obstacle detection without AI (based on lower frame region)
            self.simple_obstacle_detection(frame)

    def process_depth(self, frame):
        """Run MiDaS depth estimation and detect obstacles."""
        try:
            # Transform for model
            input_batch = self.depth_transform(frame)

            if torch.cuda.is_available():
                input_batch = input_batch.cuda()

            with torch.no_grad():
                prediction = self.depth_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy()

            # Normalize depth to 0-1
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

            # Publish depth image
            depth_uint8 = (depth * 255).astype(np.uint8)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_uint8, 'mono8')
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            depth_msg.header.frame_id = 'camera_link'
            self.depth_pub.publish(depth_msg)

            # Detect obstacles in depth
            self.detect_obstacles_from_depth(depth)

        except Exception as e:
            self.get_logger().error(f'Depth estimation failed: {e}')

    def simple_obstacle_detection(self, frame):
        """Simple obstacle detection without AI - looks for objects in lower frame."""
        h, w = frame.shape[:2]

        # Look at bottom third of frame (where obstacles would be)
        roi = frame[int(h * 0.6):, :]

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Divide into left, center, right regions
        third = w // 3
        left_edges = np.sum(edges[:, :third]) / 255
        center_edges = np.sum(edges[:, third:2*third]) / 255
        right_edges = np.sum(edges[:, 2*third:]) / 255

        # Normalize
        max_edges = max(left_edges, center_edges, right_edges, 1)
        left_obstacle = left_edges / max_edges
        center_obstacle = center_edges / max_edges
        right_obstacle = right_edges / max_edges

        # Publish obstacle scores (0 = clear, 1 = blocked)
        msg = Float32MultiArray()
        msg.data = [float(left_obstacle), float(center_obstacle), float(right_obstacle)]
        self.obstacles_pub.publish(msg)

    def detect_obstacles_from_depth(self, depth):
        """Detect obstacles from depth map."""
        h, w = depth.shape

        # Look at bottom third (close obstacles)
        roi = depth[int(h * 0.6):, :]

        # Divide into left, center, right
        third = w // 3
        left_depth = np.mean(roi[:, :third])
        center_depth = np.mean(roi[:, third:2*third])
        right_depth = np.mean(roi[:, 2*third:])

        # In MiDaS, higher values = closer
        # Convert to obstacle score (1 = blocked, 0 = clear)
        left_obstacle = left_depth
        center_obstacle = center_depth
        right_obstacle = right_depth

        msg = Float32MultiArray()
        msg.data = [float(left_obstacle), float(center_obstacle), float(right_obstacle)]
        self.obstacles_pub.publish(msg)

    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
