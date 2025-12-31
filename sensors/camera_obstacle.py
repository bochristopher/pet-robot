#!/usr/bin/env python3
"""
Local Camera Obstacle Detection
Uses OpenCV for real-time obstacle detection without API calls.
Analyzes the lower portion of the frame for obstacles at floor level.
"""

import cv2
import numpy as np
import time

class CameraObstacleDetector:
    """Real-time obstacle detection using camera and OpenCV."""

    def __init__(self, camera_index=0, width=320, height=240):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera = None
        self.last_frame = None
        self.last_result = None

        # Floor region - bottom 40% of frame where obstacles appear
        self.floor_top = int(height * 0.6)

        # Divide into 3 sectors: left, center, right
        self.sector_width = width // 3

        # Edge detection thresholds (tuned for 320x240)
        # These detect RELATIVE changes - high edge count = complex scene = obstacle
        self.edge_threshold = 800000     # Obstacle present in sector
        self.close_threshold = 2500000   # Obstacle very close (total)

    def open(self):
        """Open camera."""
        if self.camera is not None and self.camera.isOpened():
            return True

        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            return False

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        return True

    def close(self):
        """Close camera."""
        if self.camera:
            self.camera.release()
            self.camera = None

    def capture_frame(self):
        """Capture a frame from camera."""
        if not self.open():
            return None

        # Clear buffer by grabbing a few frames
        for _ in range(2):
            self.camera.grab()

        ret, frame = self.camera.read()
        if ret:
            self.last_frame = frame
            return frame
        return None

    def detect_obstacles(self):
        """
        Detect obstacles in the camera view.

        Returns dict with:
            - left_blocked: bool
            - center_blocked: bool
            - right_blocked: bool
            - obstacle_distance: 'close', 'medium', 'far', or 'clear'
            - path_clear: bool (True if center is clear)
        """
        frame = self.capture_frame()
        if frame is None:
            return None

        # Extract floor region (bottom portion of frame)
        floor_region = frame[self.floor_top:, :]

        # Convert to grayscale
        gray = cv2.cvtColor(floor_region, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Analyze each sector
        h, w = edges.shape
        sector_w = w // 3

        left_edges = np.sum(edges[:, :sector_w])
        center_edges = np.sum(edges[:, sector_w:2*sector_w])
        right_edges = np.sum(edges[:, 2*sector_w:])

        # Check each sector against thresholds
        left_blocked = left_edges > self.edge_threshold
        center_blocked = center_edges > self.edge_threshold
        right_blocked = right_edges > self.edge_threshold

        # Estimate distance based on edge density in center
        total_edges = left_edges + center_edges + right_edges
        if total_edges > self.close_threshold * 3:
            distance = 'close'
        elif total_edges > self.edge_threshold * 3:
            distance = 'medium'
        elif total_edges > self.edge_threshold:
            distance = 'far'
        else:
            distance = 'clear'

        result = {
            'left_blocked': left_blocked,
            'center_blocked': center_blocked,
            'right_blocked': right_blocked,
            'obstacle_distance': distance,
            'path_clear': not center_blocked,
            'edge_counts': {
                'left': int(left_edges),
                'center': int(center_edges),
                'right': int(right_edges)
            }
        }

        self.last_result = result
        return result

    def get_recommended_action(self, result=None):
        """Get recommended action based on detection result."""
        if result is None:
            result = self.last_result
        if result is None:
            return 'forward'

        dist = result['obstacle_distance']
        left_blocked = result['left_blocked']
        center_blocked = result['center_blocked']
        right_blocked = result['right_blocked']

        if dist == 'close':
            return 'backup'
        elif center_blocked:
            if not right_blocked:
                return 'turn_right'
            elif not left_blocked:
                return 'turn_left'
            else:
                return 'backup'
        elif dist == 'medium':
            return 'slow'
        else:
            return 'forward'


def test_camera():
    """Test the camera obstacle detection."""
    print("Testing camera obstacle detection...")

    detector = CameraObstacleDetector()

    if not detector.open():
        print("Failed to open camera!")
        return

    print("Camera opened. Testing detection...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            result = detector.detect_obstacles()
            if result:
                l = "BLOCKED" if result['left_blocked'] else "clear"
                c = "BLOCKED" if result['center_blocked'] else "clear"
                r = "BLOCKED" if result['right_blocked'] else "clear"
                dist = result['obstacle_distance']
                action = detector.get_recommended_action(result)

                print(f"\rL:{l:7} C:{c:7} R:{r:7} | Dist:{dist:6} | Action:{action:10}", end="")
            else:
                print("\rCapture failed", end="")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped")
    finally:
        detector.close()


if __name__ == "__main__":
    test_camera()
