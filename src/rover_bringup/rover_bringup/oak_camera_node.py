#!/usr/bin/env python3
"""
OAK-D Lite Camera Node using DepthAI 2.x
- RGB camera with flip correction
- Stereo depth with proper OAK-D Lite settings
- On-device MobileNet-SSD object detection (Myriad X VPU)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import urllib.request


# MobileNet-SSD labels
LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]


class OakCameraNode(Node):
    def __init__(self):
        super().__init__('oak_camera_node')

        # Parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('depth_frame_id', 'camera_depth_link')
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('flip_image', True)
        self.declare_parameter('enable_depth', True)
        self.declare_parameter('enable_detection', True)  # On-device NN
        self.declare_parameter('detection_threshold', 0.5)
        self.declare_parameter('max_depth_mm', 8000)  # 8 meters max

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value
        self.depth_frame_id = self.get_parameter('depth_frame_id').value
        self.publish_compressed = self.get_parameter('publish_compressed').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.flip_image = self.get_parameter('flip_image').value
        self.enable_depth = self.get_parameter('enable_depth').value
        self.enable_detection = self.get_parameter('enable_detection').value
        self.detection_threshold = self.get_parameter('detection_threshold').value
        self.max_depth_mm = self.get_parameter('max_depth_mm').value

        # Publishers
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        if self.publish_compressed:
            self.compressed_pub = self.create_publisher(
                CompressedImage, '/camera/image_raw/compressed', 10)

        if self.enable_depth:
            self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
            self.depth_colored_pub = self.create_publisher(
                CompressedImage, '/camera/depth/image_colored/compressed', 10)

        if self.enable_detection:
            self.detection_pub = self.create_publisher(Detection2DArray, '/camera/detections', 10)
            self.detection_image_pub = self.create_publisher(
                CompressedImage, '/camera/detections/image/compressed', 10)

        # DepthAI
        self.device = None
        self.rgb_queue = None
        self.depth_queue = None
        self.detection_queue = None
        self.camera_info_msg = None

        try:
            self._setup_pipeline()
        except Exception as e:
            self.get_logger().error(f'Failed to initialize OAK camera: {e}')
            import traceback
            traceback.print_exc()
            return

        self.create_timer(1.0 / self.fps, self.capture_and_publish)

    def _download_model(self):
        """Download MobileNet-SSD blob for Myriad X."""
        model_dir = Path.home() / '.cache' / 'depthai_models'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'mobilenet-ssd_openvino_2021.4_6shave.blob'

        if not model_path.exists():
            self.get_logger().info('Downloading MobileNet-SSD model for Myriad X...')
            url = 'https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/network/mobilenet-ssd_openvino_2021.4_6shave.blob'
            urllib.request.urlretrieve(url, model_path)
            self.get_logger().info('Model downloaded!')

        return str(model_path)

    def _setup_pipeline(self):
        """Set up DepthAI pipeline."""
        pipeline = dai.Pipeline()

        # ===== RGB Camera =====
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(self.width, self.height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(self.fps)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        # RGB output
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        # ===== Stereo Depth =====
        if self.enable_depth:
            mono_left = pipeline.create(dai.node.MonoCamera)
            mono_right = pipeline.create(dai.node.MonoCamera)
            stereo = pipeline.create(dai.node.StereoDepth)

            # OAK-D Lite uses 480P mono cameras
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
            mono_left.setCamera("left")
            mono_left.setFps(self.fps)

            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
            mono_right.setCamera("right")
            mono_right.setFps(self.fps)

            # Stereo settings optimized for OAK-D Lite
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            stereo.setLeftRightCheck(True)
            stereo.setExtendedDisparity(True)  # Better for close range
            stereo.setSubpixel(False)  # Faster
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

            # Configure depth output
            config = stereo.initialConfig.get()
            config.postProcessing.speckleFilter.enable = True
            config.postProcessing.speckleFilter.speckleRange = 50
            config.postProcessing.temporalFilter.enable = True
            config.postProcessing.spatialFilter.enable = True
            config.postProcessing.spatialFilter.holeFillingRadius = 2
            config.postProcessing.spatialFilter.numIterations = 1
            config.postProcessing.thresholdFilter.minRange = 100   # 10cm min
            config.postProcessing.thresholdFilter.maxRange = self.max_depth_mm
            stereo.initialConfig.set(config)

            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

        # ===== Neural Network (On-Device Object Detection) =====
        if self.enable_detection:
            model_path = self._download_model()

            detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
            detection_nn.setConfidenceThreshold(self.detection_threshold)
            detection_nn.setBlobPath(model_path)
            detection_nn.setNumInferenceThreads(2)
            detection_nn.input.setBlocking(False)

            # Need to resize for NN (300x300)
            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setResize(300, 300)
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
            cam_rgb.preview.link(manip.inputImage)
            manip.out.link(detection_nn.input)

            xout_nn = pipeline.create(dai.node.XLinkOut)
            xout_nn.setStreamName("detections")
            detection_nn.out.link(xout_nn.input)

        # Start device
        self.device = dai.Device(pipeline)
        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        if self.enable_depth:
            self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        if self.enable_detection:
            self.detection_queue = self.device.getOutputQueue(name="detections", maxSize=4, blocking=False)

        self._setup_camera_info()

        features = ["RGB"]
        if self.enable_depth:
            features.append("Stereo Depth")
        if self.enable_detection:
            features.append("MobileNet-SSD (Myriad X)")
        if self.flip_image:
            features.append("flipped 180°")

        self.get_logger().info(
            f'OAK camera opened: {self.device.getDeviceName()} @ {self.width}x{self.height} '
            f'{self.fps}fps [{" + ".join(features)}]'
        )

    def _setup_camera_info(self):
        """Get camera calibration."""
        try:
            calib = self.device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, self.width, self.height)

            self.camera_info_msg = CameraInfo()
            self.camera_info_msg.header.frame_id = self.frame_id
            self.camera_info_msg.width = self.width
            self.camera_info_msg.height = self.height
            self.camera_info_msg.distortion_model = 'plumb_bob'

            fx, fy = intrinsics[0][0], intrinsics[1][1]
            cx, cy = intrinsics[0][2], intrinsics[1][2]
            self.camera_info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            self.camera_info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.camera_info_msg.d = [0.0] * 5

        except Exception as e:
            self.get_logger().warn(f'Could not get calibration: {e}')

    def capture_and_publish(self):
        now = self.get_clock().now().to_msg()
        frame = None
        detections = None

        # RGB
        if self.rgb_queue:
            rgb_data = self.rgb_queue.tryGet()
            if rgb_data is not None:
                frame = rgb_data.getCvFrame()
                if self.flip_image:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                self._publish_rgb(frame, now)

        # Detections (from Myriad X)
        if self.enable_detection and self.detection_queue:
            det_data = self.detection_queue.tryGet()
            if det_data is not None:
                detections = det_data.detections
                self._publish_detections(detections, frame, now)

        # Depth
        if self.enable_depth and self.depth_queue:
            depth_data = self.depth_queue.tryGet()
            if depth_data is not None:
                depth_frame = depth_data.getFrame()
                if self.flip_image:
                    depth_frame = cv2.rotate(depth_frame, cv2.ROTATE_180)
                self._publish_depth(depth_frame, now)

    def _publish_rgb(self, frame, stamp):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = self.frame_id
            self.image_pub.publish(img_msg)

            if self.camera_info_msg:
                self.camera_info_msg.header.stamp = stamp
                self.camera_info_pub.publish(self.camera_info_msg)

            if self.publish_compressed:
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = stamp
                compressed_msg.header.frame_id = self.frame_id
                compressed_msg.format = 'jpeg'
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                compressed_msg.data = jpeg.tobytes()
                self.compressed_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f'RGB publish error: {e}')

    def _publish_depth(self, depth_frame, stamp):
        try:
            # Raw depth (16UC1 in mm)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_frame.astype(np.uint16), encoding='16UC1')
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = self.depth_frame_id
            self.depth_pub.publish(depth_msg)

            # Colorized depth - normalize to valid range only
            depth_valid = depth_frame.copy().astype(np.float32)
            depth_valid[depth_valid == 0] = np.nan  # Mark invalid as NaN

            # Get actual min/max for better visualization
            valid_mask = ~np.isnan(depth_valid)
            if np.any(valid_mask):
                min_val = np.nanmin(depth_valid)
                max_val = np.nanpercentile(depth_valid[valid_mask], 95)  # Use 95th percentile
                max_val = max(max_val, min_val + 100)  # Ensure range

                # Normalize
                depth_norm = (depth_valid - min_val) / (max_val - min_val)
                depth_norm = np.clip(depth_norm, 0, 1)
                depth_norm = np.nan_to_num(depth_norm, nan=0.0)
                depth_norm = (depth_norm * 255).astype(np.uint8)
                depth_norm[~valid_mask] = 0

                # Apply colormap (TURBO is better than JET)
                depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                depth_colored[~valid_mask] = [0, 0, 0]  # Black for invalid
            else:
                depth_colored = np.zeros((depth_frame.shape[0], depth_frame.shape[1], 3), dtype=np.uint8)

            colored_msg = CompressedImage()
            colored_msg.header.stamp = stamp
            colored_msg.header.frame_id = self.depth_frame_id
            colored_msg.format = 'jpeg'
            _, jpeg = cv2.imencode('.jpg', depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
            colored_msg.data = jpeg.tobytes()
            self.depth_colored_pub.publish(colored_msg)

        except Exception as e:
            self.get_logger().debug(f'Depth publish error: {e}')

    def _publish_detections(self, detections, frame, stamp):
        """Publish detections from on-device neural network."""
        try:
            det_array = Detection2DArray()
            det_array.header.stamp = stamp
            det_array.header.frame_id = self.frame_id

            vis_frame = frame.copy() if frame is not None else None

            for det in detections:
                # Flip bounding box if image is flipped
                if self.flip_image:
                    x1 = 1.0 - det.xmax
                    x2 = 1.0 - det.xmin
                    y1 = 1.0 - det.ymax
                    y2 = 1.0 - det.ymin
                else:
                    x1, x2 = det.xmin, det.xmax
                    y1, y2 = det.ymin, det.ymax

                # Create Detection2D message
                detection_msg = Detection2D()
                detection_msg.header = det_array.header

                # Bounding box (center + size)
                detection_msg.bbox.center.position.x = (x1 + x2) / 2.0 * self.width
                detection_msg.bbox.center.position.y = (y1 + y2) / 2.0 * self.height
                detection_msg.bbox.size_x = (x2 - x1) * self.width
                detection_msg.bbox.size_y = (y2 - y1) * self.height

                # Hypothesis
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = LABELS[det.label] if det.label < len(LABELS) else str(det.label)
                hyp.hypothesis.score = det.confidence
                detection_msg.results.append(hyp)

                det_array.detections.append(detection_msg)

                # Draw on visualization frame
                if vis_frame is not None:
                    px1, py1 = int(x1 * self.width), int(y1 * self.height)
                    px2, py2 = int(x2 * self.width), int(y2 * self.height)
                    label = LABELS[det.label] if det.label < len(LABELS) else f"class_{det.label}"
                    color = (0, 255, 0) if label == "person" else (255, 128, 0)

                    cv2.rectangle(vis_frame, (px1, py1), (px2, py2), color, 2)
                    text = f"{label}: {det.confidence:.0%}"
                    cv2.putText(vis_frame, text, (px1, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.detection_pub.publish(det_array)

            # Publish visualization
            if vis_frame is not None:
                vis_msg = CompressedImage()
                vis_msg.header.stamp = stamp
                vis_msg.header.frame_id = self.frame_id
                vis_msg.format = 'jpeg'
                _, jpeg = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                vis_msg.data = jpeg.tobytes()
                self.detection_image_pub.publish(vis_msg)

        except Exception as e:
            self.get_logger().debug(f'Detection publish error: {e}')

    def destroy_node(self):
        if self.device:
            self.device.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OakCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
