#!/usr/bin/env python3
"""
OAK-D Lite Depth Node - Nav2 Optimized
DepthAI 2.x compatible

Stripped down for autonomous navigation:
- Stereo depth only (no RGB, no neural network)
- High-performance point cloud for Nav2 costmap
- Target: 25-30+ fps
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from tf2_ros import StaticTransformBroadcaster
import depthai as dai
import numpy as np
import time


class OakDepthNode(Node):
    def __init__(self):
        super().__init__('oak_depth_node')

        # ===== Parameters =====
        self.declare_parameter('point_cloud_decimation', 4)
        self.declare_parameter('min_depth_mm', 200)
        self.declare_parameter('max_depth_mm', 4000)
        self.declare_parameter('publish_depth_image', False)
        self.declare_parameter('target_fps', 30)
        self.declare_parameter('width', 416)
        self.declare_parameter('height', 240)
        self.declare_parameter('flip_image', True)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('depth_optical_frame', 'camera_depth_optical_frame')

        self.decimation = self.get_parameter('point_cloud_decimation').value
        self.min_depth_mm = self.get_parameter('min_depth_mm').value
        self.max_depth_mm = self.get_parameter('max_depth_mm').value
        self.publish_depth_image = self.get_parameter('publish_depth_image').value
        self.target_fps = self.get_parameter('target_fps').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.flip_image = self.get_parameter('flip_image').value
        self.frame_id = self.get_parameter('frame_id').value
        self.depth_optical_frame = self.get_parameter('depth_optical_frame').value

        # ===== QoS for Nav2 =====
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ===== Publishers =====
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/camera/points', sensor_qos)

        if self.publish_depth_image:
            self.depth_pub = self.create_publisher(
                Image, '/camera/depth/image_raw', sensor_qos)
            self.depth_info_pub = self.create_publisher(
                CameraInfo, '/camera/depth/camera_info', sensor_qos)
            self.bridge = CvBridge()

        # ===== TF Broadcaster =====
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        # ===== Camera intrinsics (will be updated from device) =====
        self.fx = self.fy = 200.0  # Will be updated
        self.cx = self.width / 2
        self.cy = self.height / 2

        # ===== Precompute pixel coordinate grids for point cloud =====
        self._precompute_grids()

        # ===== DepthAI =====
        self.device = None
        self.depth_queue = None
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_report_interval = 100

        try:
            self._setup_and_start()
        except Exception as e:
            self.get_logger().error(f'Failed to initialize OAK camera: {e}')
            import traceback
            traceback.print_exc()
            return

        # Main loop - run as fast as possible
        self.create_timer(1.0 / (self.target_fps + 10), self.process_depth)

        # Watchdog
        self.create_timer(10.0, self._check_connection)

    def _precompute_grids(self):
        """Precompute pixel coordinate grids for fast point cloud generation."""
        dec = self.decimation
        h, w = self.height, self.width

        # Decimated dimensions
        self.h_dec = h // dec
        self.w_dec = w // dec

        # Create meshgrid of pixel coordinates (decimated)
        u = np.arange(0, w, dec, dtype=np.float32)
        v = np.arange(0, h, dec, dtype=np.float32)
        self.u_grid, self.v_grid = np.meshgrid(u, v)

    def _update_intrinsics_grid(self):
        """Update the precomputed grids with actual camera intrinsics."""
        # Normalized coordinates for 3D projection
        self.x_factor = (self.u_grid - self.cx) / self.fx
        self.y_factor = (self.v_grid - self.cy) / self.fy

    def _publish_static_transforms(self):
        """Publish static TF frames for camera."""
        transforms = []
        now = self.get_clock().now().to_msg()

        # camera_link -> camera_depth_optical_frame
        # Optical frame: Z forward, X right, Y down
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.frame_id
        t.child_frame_id = self.depth_optical_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        # Rotation from ROS convention (X forward) to optical (Z forward)
        t.transform.rotation.x = -0.5
        t.transform.rotation.y = 0.5
        t.transform.rotation.z = -0.5
        t.transform.rotation.w = 0.5
        transforms.append(t)

        self.tf_broadcaster.sendTransform(transforms)
        self.get_logger().info(f'Published TF: {self.frame_id} -> {self.depth_optical_frame}')

    def _setup_and_start(self):
        """Set up DepthAI pipeline and start device."""
        pipeline = dai.Pipeline()

        # ===== Mono Cameras =====
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)

        # Use 400P for speed (400x640 -> we'll output at configured resolution)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setCamera("left")
        mono_left.setFps(self.target_fps)

        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setCamera("right")
        mono_right.setFps(self.target_fps)

        # ===== Stereo Depth =====
        stereo = pipeline.create(dai.node.StereoDepth)

        # Performance-optimized settings
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setSubpixel(False)  # Faster
        stereo.setLeftRightCheck(True)  # Filter bad matches
        stereo.setExtendedDisparity(False)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.setOutputSize(self.width, self.height)

        # Post-processing (minimal for speed)
        config = stereo.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.temporalFilter.alpha = 0.4
        config.postProcessing.spatialFilter.enable = False  # Skip for speed
        config.postProcessing.thresholdFilter.minRange = self.min_depth_mm
        config.postProcessing.thresholdFilter.maxRange = self.max_depth_mm
        stereo.initialConfig.set(config)

        # Link mono to stereo
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Depth output
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # Start device
        self.device = dai.Device(pipeline)
        self.depth_queue = self.device.getOutputQueue(
            name="depth", maxSize=2, blocking=False)

        # Get camera calibration
        self._setup_camera_info()

        self.get_logger().info(
            f'OAK-D Lite started: {self.width}x{self.height} @ {self.target_fps}fps | '
            f'Decimation: {self.decimation}x ({self.w_dec}x{self.h_dec} points) | '
            f'Range: {self.min_depth_mm/1000:.1f}-{self.max_depth_mm/1000:.1f}m'
        )

    def _setup_camera_info(self):
        """Get camera intrinsics from device calibration."""
        try:
            calib = self.device.readCalibration()
            # Get intrinsics for the depth output resolution
            intrinsics = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, self.width, self.height)

            self.fx = intrinsics[0][0]
            self.fy = intrinsics[1][1]
            self.cx = intrinsics[0][2]
            self.cy = intrinsics[1][2]

            self.get_logger().info(
                f'Camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx:.1f} cy={self.cy:.1f}'
            )

            # Update precomputed grids with actual intrinsics
            self._update_intrinsics_grid()

            # Store for CameraInfo message
            if self.publish_depth_image:
                self.camera_info_msg = CameraInfo()
                self.camera_info_msg.width = self.width
                self.camera_info_msg.height = self.height
                self.camera_info_msg.distortion_model = 'plumb_bob'
                self.camera_info_msg.k = [
                    self.fx, 0.0, self.cx,
                    0.0, self.fy, self.cy,
                    0.0, 0.0, 1.0
                ]
                self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                self.camera_info_msg.p = [
                    self.fx, 0.0, self.cx, 0.0,
                    0.0, self.fy, self.cy, 0.0,
                    0.0, 0.0, 1.0, 0.0
                ]
                self.camera_info_msg.d = [0.0] * 5

        except Exception as e:
            self.get_logger().warn(f'Could not get calibration: {e}, using defaults')
            self._update_intrinsics_grid()

    def _check_connection(self):
        """Watchdog to detect camera disconnection."""
        if time.time() - self.last_frame_time > 10.0:
            self.get_logger().warn('No depth frames for 10s, camera may be disconnected')

    def process_depth(self):
        """Process depth frame and publish point cloud."""
        if self.depth_queue is None:
            return

        depth_data = self.depth_queue.tryGet()
        if depth_data is None:
            return

        depth_frame = depth_data.getFrame()
        self.last_frame_time = time.time()

        # Flip if camera is mounted upside down
        if self.flip_image:
            depth_frame = np.rot90(depth_frame, 2)

        now = self.get_clock().now().to_msg()

        # Publish point cloud (always)
        self._publish_pointcloud(depth_frame, now)

        # Publish depth image (optional, for debugging)
        if self.publish_depth_image:
            self._publish_depth_image(depth_frame, now)

        # FPS reporting
        self.frame_count += 1
        if self.frame_count % self.fps_report_interval == 0:
            elapsed = time.time() - self.last_frame_time + 0.001
            # This isn't quite right but gives rough idea
            self.get_logger().info(f'Processed {self.frame_count} frames')

    def _publish_pointcloud(self, depth_frame, stamp):
        """Publish decimated point cloud for Nav2."""
        dec = self.decimation

        # Decimate depth frame
        depth_dec = depth_frame[::dec, ::dec].astype(np.float32)

        # Convert depth from mm to meters
        z = depth_dec / 1000.0

        # Valid depth mask (within range)
        min_z = self.min_depth_mm / 1000.0
        max_z = self.max_depth_mm / 1000.0
        valid = (z > min_z) & (z < max_z)

        # Calculate 3D coordinates using precomputed grids
        x = self.x_factor * z
        y = self.y_factor * z

        # Stack into Nx3 array and filter valid points
        points = np.stack([x, y, z], axis=-1)
        points = points[valid]

        if len(points) == 0:
            return

        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.depth_optical_frame
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False

        # XYZ fields only (12 bytes per point)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.data = points.astype(np.float32).tobytes()

        self.pointcloud_pub.publish(msg)

    def _publish_depth_image(self, depth_frame, stamp):
        """Publish depth image for debugging."""
        depth_msg = self.bridge.cv2_to_imgmsg(
            depth_frame.astype(np.uint16), encoding='16UC1')
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.depth_optical_frame
        self.depth_pub.publish(depth_msg)

        # Camera info
        self.camera_info_msg.header.stamp = stamp
        self.camera_info_msg.header.frame_id = self.depth_optical_frame
        self.depth_info_pub.publish(self.camera_info_msg)

    def destroy_node(self):
        """Clean shutdown."""
        if self.device:
            self.device.close()
            self.get_logger().info('OAK camera closed')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OakDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
