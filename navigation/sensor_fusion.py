#!/usr/bin/env python3
"""
Sensor Fusion Module
Combines LiDAR and Camera for more reliable obstacle detection.

Strategy:
- LiDAR runs continuously (fast, free)
- Camera only activates when LiDAR readings are suspicious:
  - Few points in a zone (possible reflective surface like glass/mirror)
  - All zones report very far distances (sanity check)
- Camera confirms: "Is there actually something there?"
- Fusion result is used for navigation decisions

This approach gives:
- Fast navigation (LiDAR-based, no delays normally)
- Low API cost (camera only when needed)
- Catches glass/mirrors the LiDAR misses
"""

import time
import sys
from typing import Optional
from dataclasses import dataclass

sys.path.insert(0, '/home/bo/robot_pet')
from perception.lidar_detector import LidarScan, LidarDetector
from perception.openai_vision import OpenAIVision


@dataclass
class FusionResult:
    """Combined sensor reading."""
    front_clear: bool
    front_distance: float
    left_clear: bool
    left_distance: float
    right_clear: bool
    right_distance: float
    recommended_action: str  # "forward", "left", "right", "stop", "backup"
    confidence: float  # 0.0 - 1.0
    camera_used: bool  # True if camera was consulted
    camera_override: bool  # True if camera changed the decision


class SensorFusion:
    """
    Fuses LiDAR and Camera data for reliable navigation.

    Usage:
        fusion = SensorFusion()
        result = fusion.get_fused_reading(lidar_scan)
        if result.front_clear:
            # Safe to move forward
    """

    def __init__(self, vision: Optional[OpenAIVision] = None):
        self.vision = vision
        self.camera_checks = 0
        self.camera_overrides = 0
        self.last_camera_check = 0
        self.camera_cooldown = 2.0  # Seconds between camera checks

        # Thresholds for triggering camera check
        self.suspicious_distance = 3.0  # If all zones > 3m, sanity check
        self.min_confidence = 0.7  # Minimum camera confidence to trust

        print("[Fusion] ✅ Sensor fusion initialized")
        if vision:
            print("[Fusion]    Camera validation: ENABLED")
        else:
            print("[Fusion]    Camera validation: DISABLED (no vision module)")

    def should_check_camera(self, scan: LidarScan) -> bool:
        """Determine if camera should validate LiDAR reading."""
        # No vision module available
        if not self.vision:
            return False

        # Respect cooldown to avoid spamming camera
        if time.time() - self.last_camera_check < self.camera_cooldown:
            return False

        # Check if any zone is suspicious (few points = possible reflective surface)
        if scan.front.suspicious or scan.left.suspicious or scan.right.suspicious:
            return True

        # Sanity check: if everything looks very clear, verify with camera
        if (scan.front.min_distance > self.suspicious_distance and
            scan.left.min_distance > self.suspicious_distance and
            scan.right.min_distance > self.suspicious_distance):
            return True

        return False

    def get_fused_reading(self, scan: LidarScan) -> FusionResult:
        """
        Get a fused sensor reading.

        Uses LiDAR as primary, camera for validation when suspicious.
        """
        # Start with LiDAR data
        front_clear = scan.front.clear
        front_distance = scan.front.min_distance
        left_clear = scan.left.clear
        left_distance = scan.left.min_distance
        right_clear = scan.right.clear
        right_distance = scan.right.min_distance

        camera_used = False
        camera_override = False
        confidence = 0.9 if not scan.front.suspicious else 0.6

        # Determine initial action based on LiDAR
        if not front_clear or front_distance < 0.5:
            if left_distance > right_distance:
                recommended_action = "left"
            else:
                recommended_action = "right"
        elif front_distance < 1.0:
            recommended_action = "forward"  # Slow forward
        else:
            recommended_action = "forward"

        # Check if camera validation is needed
        if self.should_check_camera(scan):
            camera_result = self._check_camera()
            camera_used = True
            self.camera_checks += 1
            self.last_camera_check = time.time()

            if camera_result:
                # Camera saw something - evaluate
                cam_clear = camera_result.get("path_clear", True)
                cam_action = camera_result.get("recommended_action", "forward")
                cam_confidence = camera_result.get("confidence", 0.5)

                # If camera is confident and disagrees with LiDAR, override
                if cam_confidence >= self.min_confidence:
                    if not cam_clear and front_clear:
                        # Camera sees obstacle that LiDAR missed!
                        print(f"[Fusion] ⚠️ Camera override: obstacle detected!")
                        front_clear = False
                        front_distance = min(front_distance, 0.8)  # Assume close
                        recommended_action = cam_action
                        camera_override = True
                        self.camera_overrides += 1
                        confidence = cam_confidence
                    elif cam_clear and not front_clear:
                        # Camera confirms clear - trust camera more
                        confidence = cam_confidence

        return FusionResult(
            front_clear=front_clear,
            front_distance=front_distance,
            left_clear=left_clear,
            left_distance=left_distance,
            right_clear=right_clear,
            right_distance=right_distance,
            recommended_action=recommended_action,
            confidence=confidence,
            camera_used=camera_used,
            camera_override=camera_override
        )

    def _check_camera(self) -> Optional[dict]:
        """Get obstacle detection from camera."""
        if not self.vision:
            return None

        try:
            result = self.vision.detect_obstacles()
            if result:
                print(f"[Fusion] 📷 Camera: path_clear={result.get('path_clear')}, "
                      f"confidence={result.get('confidence', 0):.1f}")
            return result
        except Exception as e:
            print(f"[Fusion] ❌ Camera error: {e}")
            return None

    def get_stats(self) -> dict:
        """Get fusion statistics."""
        return {
            "camera_checks": self.camera_checks,
            "camera_overrides": self.camera_overrides,
            "override_rate": (self.camera_overrides / self.camera_checks * 100
                             if self.camera_checks > 0 else 0)
        }


# Convenience function
def create_fusion(enable_camera: bool = True) -> SensorFusion:
    """Create a sensor fusion instance."""
    vision = None
    if enable_camera:
        try:
            vision = OpenAIVision()
        except Exception as e:
            print(f"[Fusion] ⚠️ Camera disabled: {e}")

    return SensorFusion(vision=vision)


if __name__ == "__main__":
    # Test sensor fusion
    print("\n" + "="*60)
    print("🔀 Sensor Fusion Test")
    print("="*60)

    fusion = create_fusion(enable_camera=True)
    lidar = LidarDetector()

    if not lidar.start():
        print("Failed to start LiDAR")
        exit(1)

    try:
        for i in range(5):
            scan = lidar.get_scan()
            if scan:
                result = fusion.get_fused_reading(scan)

                print(f"\n📊 Reading #{i+1}")
                print(f"   Front: {result.front_distance:.2f}m "
                      f"({'clear' if result.front_clear else 'blocked'})")
                print(f"   Left:  {result.left_distance:.2f}m")
                print(f"   Right: {result.right_distance:.2f}m")
                print(f"   Action: {result.recommended_action}")
                print(f"   Confidence: {result.confidence:.0%}")
                if result.camera_used:
                    print(f"   📷 Camera: {'OVERRIDE' if result.camera_override else 'confirmed'}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    finally:
        lidar.stop()
        print("\n" + "="*60)
        stats = fusion.get_stats()
        print(f"📊 Fusion Stats:")
        print(f"   Camera checks: {stats['camera_checks']}")
        print(f"   Camera overrides: {stats['camera_overrides']}")
        print("="*60)
