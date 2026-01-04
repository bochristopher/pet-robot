#!/usr/bin/env python3
"""
Camera Node using OpenCV
Works better on Jetson than v4l2_camera for memory mapping.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Parameters
        self.declare_parameter('device_id', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('publish_compressed', True)

        device_id = self.get_parameter('device_id').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_compressed = self.get_parameter('publish_compressed').value

        # OpenCV capture
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {device_id}')
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'Camera opened: {actual_w}x{actual_h} @ {fps}fps')

        # Publishers
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)

        if self.publish_compressed:
            self.compressed_pub = self.create_publisher(
                CompressedImage, '/camera/image_raw/compressed', 10)

        # Timer
        self.create_timer(1.0 / fps, self.capture_and_publish)

    def capture_and_publish(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().debug('Failed to capture frame')
            return

        now = self.get_clock().now().to_msg()

        # Publish raw image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = now
            img_msg.header.frame_id = self.frame_id
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

        # Publish compressed
        if self.publish_compressed:
            try:
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = now
                compressed_msg.header.frame_id = self.frame_id
                compressed_msg.format = 'jpeg'
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                compressed_msg.data = jpeg.tobytes()
                self.compressed_pub.publish(compressed_msg)
            except Exception as e:
                self.get_logger().debug(f'Failed to compress image: {e}')

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
