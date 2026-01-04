#!/usr/bin/env python3
"""
Object Detection Node using YOLO
Provides semantic understanding of the environment.
Publishes detected objects and can mark obstacles in costmap.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

# Try to import YOLO, fall back to basic detection if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")

        # Parameters
        self.declare_parameter("model", "yolov8n.pt")  # nano model for speed
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("device", "0")  # GPU 0
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("obstacle_classes", ["person", "chair", "dog", "cat", "backpack", "suitcase"])

        model_path = self.get_parameter("model").value
        self.conf_threshold = self.get_parameter("confidence_threshold").value
        device = self.get_parameter("device").value
        self.publish_annotated = self.get_parameter("publish_annotated").value
        self.obstacle_classes = self.get_parameter("obstacle_classes").value

        self.bridge = CvBridge()
        self.model = None

        # Load YOLO model
        if YOLO_AVAILABLE:
            try:
                self.get_logger().info(f"Loading YOLO model: {model_path}")
                self.model = YOLO(model_path)
                # Warm up the model
                self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)
                self.get_logger().info("YOLO model loaded successfully")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO: {e}")
                self.model = None
        else:
            self.get_logger().warn("YOLO not available. Install with: pip install ultralytics")

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )

        # Publishers
        self.detections_pub = self.create_publisher(
            Detection2DArray, "/detections", 10
        )
        self.objects_pub = self.create_publisher(
            String, "/detected_objects", 10
        )
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image, "/camera/image_annotated", 10
            )

        # For obstacle integration
        self.obstacle_pub = self.create_publisher(
            PointStamped, "/detected_obstacle", 10
        )

        self.get_logger().info("Object detector initialized")
        self.get_logger().info(f"Obstacle classes: {self.obstacle_classes}")

    def image_callback(self, msg: Image):
        if self.model is None:
            return

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # Run YOLO inference
        try:
            results = self.model.predict(
                cv_image,
                conf=self.conf_threshold,
                verbose=False,
                device=0
            )
        except Exception as e:
            self.get_logger().debug(f"Inference error: {e}")
            return

        # Process results
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        detected_names = []
        
        img_height, img_width = cv_image.shape[:2]
        img_center_x = img_width / 2

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # Get detection info
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                detected_names.append(f"{cls_name}:{conf:.2f}")

                # Create Detection2D message
                det = Detection2D()
                det.header = msg.header
                
                # Bounding box center and size
                det.bbox.center.position.x = (x1 + x2) / 2
                det.bbox.center.position.y = (y1 + y2) / 2
                det.bbox.size_x = x2 - x1
                det.bbox.size_y = y2 - y1

                # Object hypothesis
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = cls_name
                hyp.hypothesis.score = conf
                det.results.append(hyp)

                detection_array.detections.append(det)

                # Check if this is an obstacle class and in front of robot
                if cls_name in self.obstacle_classes:
                    box_center_x = (x1 + x2) / 2
                    box_width = x2 - x1
                    
                    # If object is in center third of image and large enough
                    if abs(box_center_x - img_center_x) < img_width / 3:
                        if box_width > img_width * 0.15:  # >15% of image width
                            # Estimate distance (rough: larger = closer)
                            relative_size = box_width / img_width
                            
                            # Publish obstacle warning
                            obstacle_msg = PointStamped()
                            obstacle_msg.header = msg.header
                            obstacle_msg.point.x = 1.0 / relative_size  # rough distance estimate
                            obstacle_msg.point.y = (box_center_x - img_center_x) / img_width
                            obstacle_msg.point.z = 0.0
                            self.obstacle_pub.publish(obstacle_msg)
                            
                            self.get_logger().info(
                                f"OBSTACLE: {cls_name} detected ahead (size: {relative_size:.1%})"
                            )

        # Publish detections
        self.detections_pub.publish(detection_array)

        # Publish object names as string
        if detected_names:
            obj_msg = String()
            obj_msg.data = ", ".join(detected_names)
            self.objects_pub.publish(obj_msg)

        # Publish annotated image
        if self.publish_annotated and len(results) > 0:
            annotated = results[0].plot()
            try:
                ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                ann_msg.header = msg.header
                self.annotated_pub.publish(ann_msg)
            except Exception as e:
                self.get_logger().debug(f"Failed to publish annotated: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
