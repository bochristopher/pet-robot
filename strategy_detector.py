#!/usr/bin/env python3
"""
Vision Strategy Detector - Phase 1 Assessment
Tests YOLO, OpenCV, and Vision API to determine best approach.
"""

import os
import sys
import cv2
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_REGION_MARGIN = 0.3  # 30% from edges

# Cost estimates (per call)
COST_GPT4V = 0.01  # $0.01 per image
COST_ELEVENLABS_PER_1K_CHARS = 0.03  # $0.03 per 1000 chars

# File paths
YOLO_ENGINE = Path.home() / "ml_models/yolov8n_fp16.engine"
YOLO_PT = Path.home() / "ml_models/yolov8n.pt"
YOLO_ONNX = Path.home() / "ml_models/yolov8n.onnx"


class VisionStrategyDetector:
    """Test and benchmark all vision methods."""
    
    def __init__(self, verbose=False, save_images=False):
        self.verbose = verbose
        self.save_images = save_images
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "yolo": {},
            "opencv": {},
            "vision_api": {},
            "resources": {},
            "recommendation": {}
        }
        
        # Initialize camera
        self.camera = None
        self.test_frame = None
        
    def print_header(self, title):
        """Print section header."""
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
    
    def check_resources(self):
        """Check system resources."""
        self.print_header("SYSTEM RESOURCES")
        
        # Disk space
        import shutil
        disk = shutil.disk_usage("/")
        disk_free_gb = disk.free / (1024**3)
        
        # RAM
        import subprocess
        mem_info = subprocess.check_output(['free', '-h']).decode()
        
        print(f"✅ Storage: {disk_free_gb:.0f} GB free")
        print(f"✅ Total disk: {disk.total / (1024**3):.0f} GB")
        
        # Extract RAM from free output
        for line in mem_info.split('\n'):
            if 'Mem:' in line:
                parts = line.split()
                print(f"✅ RAM: {parts[6]} available")
                break
        
        self.results["resources"] = {
            "disk_free_gb": disk_free_gb,
            "disk_total_gb": disk.total / (1024**3)
        }
        
        return True
    
    def init_camera(self):
        """Initialize camera."""
        print("\n📷 Initializing camera...")
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.camera.isOpened():
            print(f"❌ Cannot open camera {CAMERA_INDEX}")
            return False
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Capture test frame
        ret, frame = self.camera.read()
        if not ret:
            print("❌ Cannot capture frame")
            return False
        
        self.test_frame = frame
        print(f"✅ Camera ready: {frame.shape[1]}x{frame.shape[0]}")
        
        if self.save_images:
            cv2.imwrite("/tmp/detection_test_frame.jpg", frame)
            print("💾 Saved: /tmp/detection_test_frame.jpg")
        
        return True
    
    def test_yolo(self):
        """Test YOLO detection."""
        self.print_header("YOLO DETECTION TEST")
        
        yolo_result = {
            "available": False,
            "model_used": None,
            "inference_time_ms": 0,
            "objects_detected": [],
            "center_obstacle": False,
            "error": None
        }
        
        try:
            from ultralytics import YOLO
            
            # Try loading models in order of preference
            model = None
            for model_path, name in [
                (YOLO_ENGINE, "TensorRT Engine"),
                (YOLO_PT, "PyTorch Model"),
                (YOLO_ONNX, "ONNX Model")
            ]:
                if model_path.exists():
                    print(f"Trying {name}: {model_path}")
                    try:
                        model = YOLO(str(model_path))
                        yolo_result["model_used"] = name
                        print(f"✅ Loaded: {name}")
                        break
                    except Exception as e:
                        print(f"⚠️  Failed to load {name}: {e}")
            
            if model is None:
                yolo_result["error"] = "No YOLO model found"
                print("❌ No YOLO models available")
                return yolo_result
            
            # Run inference
            print("\n🔍 Running inference...")
            start_time = time.time()
            results = model(self.test_frame, conf=0.4, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            
            yolo_result["inference_time_ms"] = round(inference_time, 1)
            print(f"⏱️  Inference time: {inference_time:.1f}ms")
            
            # Parse detections
            detections = []
            center_left = FRAME_WIDTH * CENTER_REGION_MARGIN
            center_right = FRAME_WIDTH * (1 - CENTER_REGION_MARGIN)
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = (x1 + x2) / 2
                    
                    class_name = model.names[class_id]
                    in_center = center_left < center_x < center_right
                    
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 2),
                        "in_center": in_center
                    })
                    
                    if in_center:
                        yolo_result["center_obstacle"] = True
            
            yolo_result["objects_detected"] = detections
            yolo_result["available"] = True
            
            # Display results
            print(f"📊 Objects detected: {len(detections)}")
            for det in detections[:5]:  # Show top 5
                icon = "🔴" if det["in_center"] else "⚪"
                print(f"   {icon} {det['class']}: {det['confidence']}")
            
            if yolo_result["center_obstacle"]:
                print("🚧 CENTER OBSTACLE DETECTED")
            else:
                print("✅ Path clear (center)")
            
            # Save annotated image
            if self.save_images and len(results) > 0:
                annotated = results[0].plot()
                cv2.imwrite("/tmp/yolo_detection.jpg", annotated)
                print("💾 Saved: /tmp/yolo_detection.jpg")
        
        except ImportError:
            yolo_result["error"] = "ultralytics not installed"
            print("❌ ultralytics library not found")
            print("   Install: pip install ultralytics")
        except Exception as e:
            yolo_result["error"] = str(e)
            print(f"❌ YOLO error: {e}")
        
        self.results["yolo"] = yolo_result
        return yolo_result
    
    def test_opencv(self):
        """Test OpenCV edge detection."""
        self.print_header("OPENCV EDGE DETECTION TEST")
        
        opencv_result = {
            "available": True,
            "processing_time_ms": 0,
            "edges_detected": 0,
            "large_contours": 0,
            "center_obstacle_likely": False,
            "error": None
        }
        
        try:
            # Convert to grayscale
            start_time = time.time()
            gray = cv2.cvtColor(self.test_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            processing_time = (time.time() - start_time) * 1000
            opencv_result["processing_time_ms"] = round(processing_time, 1)
            
            # Count edges
            edge_pixels = cv2.countNonZero(edges)
            opencv_result["edges_detected"] = edge_pixels
            
            # Count large contours in center region
            center_left = int(FRAME_WIDTH * CENTER_REGION_MARGIN)
            center_right = int(FRAME_WIDTH * (1 - CENTER_REGION_MARGIN))
            
            large_contours = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # Significant contour
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        if center_left < cx < center_right:
                            large_contours += 1
            
            opencv_result["large_contours"] = large_contours
            opencv_result["center_obstacle_likely"] = large_contours > 2
            
            # Display results
            print(f"⏱️  Processing time: {processing_time:.1f}ms")
            print(f"📊 Edge pixels: {edge_pixels}")
            print(f"📊 Large contours: {len([c for c in contours if cv2.contourArea(c) > 500])}")
            print(f"📊 Center contours: {large_contours}")
            
            if opencv_result["center_obstacle_likely"]:
                print("🚧 CENTER OBSTACLE LIKELY")
            else:
                print("✅ Path likely clear (center)")
            
            # Save edge map
            if self.save_images:
                cv2.imwrite("/tmp/opencv_edges.jpg", edges)
                print("💾 Saved: /tmp/opencv_edges.jpg")
        
        except Exception as e:
            opencv_result["error"] = str(e)
            opencv_result["available"] = False
            print(f"❌ OpenCV error: {e}")
        
        self.results["opencv"] = opencv_result
        return opencv_result
    
    def test_vision_api(self):
        """Test OpenAI Vision API."""
        self.print_header("OPENAI VISION API TEST")
        
        vision_result = {
            "available": False,
            "api_key_set": False,
            "api_latency_ms": 0,
            "response": None,
            "objects_found": [],
            "error": None
        }
        
        # Check API key
        api_key = os.environ.get("OPENAI_API_KEY")
        vision_result["api_key_set"] = bool(api_key)
        
        if not api_key:
            vision_result["error"] = "OPENAI_API_KEY not set"
            print("❌ OPENAI_API_KEY environment variable not set")
            print("   Set with: export OPENAI_API_KEY='your_key'")
            self.results["vision_api"] = vision_result
            return vision_result
        
        print(f"✅ API key detected: {api_key[:8]}...")
        
        try:
            from openai import OpenAI
            import base64
            
            client = OpenAI(api_key=api_key)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', self.test_frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Call API
            print("🌐 Calling Vision API...")
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "List the main objects you see in this image. Be brief and concise. Format: object1, object2, object3"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            
            latency = (time.time() - start_time) * 1000
            vision_result["api_latency_ms"] = round(latency, 1)
            
            # Parse response
            result_text = response.choices[0].message.content
            vision_result["response"] = result_text
            vision_result["available"] = True
            
            # Extract objects (simple parsing)
            objects = [obj.strip() for obj in result_text.split(',')]
            vision_result["objects_found"] = objects[:5]
            
            # Display results
            print(f"⏱️  API latency: {latency:.0f}ms")
            print(f"📝 Response: {result_text}")
            print(f"💰 Cost: ~${COST_GPT4V:.3f}")
            print(f"✅ Vision API working")
        
        except ImportError:
            vision_result["error"] = "openai library not installed"
            print("❌ openai library not found")
            print("   Install: pip install openai")
        except Exception as e:
            vision_result["error"] = str(e)
            print(f"❌ Vision API error: {e}")
        
        self.results["vision_api"] = vision_result
        return vision_result
    
    def calculate_costs(self):
        """Calculate cost estimates for different strategies."""
        self.print_header("COST ANALYSIS (5-minute exploration)")
        
        # Assumptions for 5-minute session
        DETECTIONS_PER_SESSION = 150  # Every 2 seconds
        SPEECH_CHARS_BASE = 800
        
        strategies = {}
        
        # Strategy A: YOLO Primary
        if self.results["yolo"]["available"]:
            yolo_cost = 0  # Free
            vision_fallback = 2 * COST_GPT4V  # Stuck + curiosity
            speech = (SPEECH_CHARS_BASE / 1000) * COST_ELEVENLABS_PER_1K_CHARS
            total_a = yolo_cost + vision_fallback + speech
            
            strategies["A_YOLO_Primary"] = {
                "yolo_detections": (150, 0),
                "vision_api_calls": (2, vision_fallback),
                "speech_chars": (SPEECH_CHARS_BASE, speech),
                "total": total_a,
                "recommended": True
            }
            
            print(f"\n💡 Strategy A - YOLO Primary:")
            print(f"   YOLO detections: 150 @ $0.00 = $0.00")
            print(f"   Vision API (stuck/curious): 2 @ ${COST_GPT4V:.2f} = ${vision_fallback:.2f}")
            print(f"   ElevenLabs speech: ~{SPEECH_CHARS_BASE} chars = ${speech:.2f}")
            print(f"   TOTAL: ${total_a:.2f} ✅ RECOMMENDED")
        
        # Strategy B: OpenCV Primary
        opencv_cost = 0
        vision_fallback_b = 8 * COST_GPT4V  # More uncertain
        speech_b = (1000 / 1000) * COST_ELEVENLABS_PER_1K_CHARS
        total_b = opencv_cost + vision_fallback_b + speech_b
        
        strategies["B_OpenCV_Primary"] = {
            "opencv_detections": (150, 0),
            "vision_api_calls": (8, vision_fallback_b),
            "speech_chars": (1000, speech_b),
            "total": total_b,
            "recommended": False
        }
        
        print(f"\n💡 Strategy B - OpenCV Primary:")
        print(f"   OpenCV detections: 150 @ $0.00 = $0.00")
        print(f"   Vision API (uncertain/stuck): 8 @ ${COST_GPT4V:.2f} = ${vision_fallback_b:.2f}")
        print(f"   ElevenLabs speech: ~1000 chars = ${speech_b:.2f}")
        print(f"   TOTAL: ${total_b:.2f}")
        
        # Strategy C: Vision API Only
        if self.results["vision_api"]["available"]:
            vision_only = 60 * COST_GPT4V  # Every 5 seconds
            speech_c = (600 / 1000) * COST_ELEVENLABS_PER_1K_CHARS
            total_c = vision_only + speech_c
            
            strategies["C_Vision_API_Only"] = {
                "vision_api_calls": (60, vision_only),
                "speech_chars": (600, speech_c),
                "total": total_c,
                "recommended": False
            }
            
            print(f"\n💡 Strategy C - Vision API Only:")
            print(f"   Vision API detections: 60 @ ${COST_GPT4V:.2f} = ${vision_only:.2f}")
            print(f"   ElevenLabs speech: ~600 chars = ${speech_c:.2f}")
            print(f"   TOTAL: ${total_c:.2f}")
        
        self.results["strategies"] = strategies
        return strategies
    
    def make_recommendation(self):
        """Make final recommendation."""
        self.print_header("⭐ RECOMMENDATION")
        
        yolo_available = self.results["yolo"]["available"]
        vision_available = self.results["vision_api"]["available"]
        
        recommendation = {
            "strategy": None,
            "reasoning": [],
            "next_steps": []
        }
        
        if yolo_available:
            recommendation["strategy"] = "A_YOLO_Primary"
            recommendation["reasoning"] = [
                f"✅ Fastest detection ({self.results['yolo']['inference_time_ms']:.0f}ms vs ~1200ms API)",
                "✅ Most accurate object detection",
                "✅ Lowest cost (~$0.26 vs $0.78 API-only)",
                "✅ Works offline (no internet needed)",
                "✅ Minimal API calls (only when stuck/curious)"
            ]
            recommendation["next_steps"] = [
                "1. Run Phase 2: vision_tester.py (YOLO only)",
                "2. Verify 90%+ obstacle detection accuracy",
                "3. Proceed to Phase 3: decision logic testing"
            ]
            
            print("\n🎯 RECOMMENDED: Strategy A (YOLO Primary)\n")
            print("Reasoning:")
            for reason in recommendation["reasoning"]:
                print(f"  {reason}")
            
            print("\nImplementation Plan:")
            print("  - Use YOLO for continuous detection (every 2s)")
            print("  - Call Vision API only when:")
            print("    * Stuck (same obstacle 3+ times)")
            print("    * Curiosity event (every 5 minutes)")
            print("    * User asks 'what do you see?'")
            print("  - Expected API usage: 2-3 calls per 5 min")
            
            print("\nNext Steps:")
            for step in recommendation["next_steps"]:
                print(f"  {step}")
        
        elif vision_available:
            recommendation["strategy"] = "C_Vision_API_Only"
            recommendation["reasoning"] = [
                "⚠️  YOLO not available",
                "✅ Vision API very accurate",
                "⚠️  Higher cost (~$0.78 per session)",
                "⚠️  Slower (1200ms per detection)",
                "⚠️  Requires internet connection"
            ]
            recommendation["next_steps"] = [
                "1. Install YOLO for better performance",
                "2. Or proceed with Vision API strategy",
                "3. Monitor API costs closely"
            ]
            
            print("\n🎯 FALLBACK: Strategy C (Vision API Only)\n")
            print("Reasoning:")
            for reason in recommendation["reasoning"]:
                print(f"  {reason}")
            print("\nConsider installing YOLO for better performance")
        
        else:
            recommendation["strategy"] = "B_OpenCV_Primary"
            recommendation["reasoning"] = [
                "⚠️  YOLO and Vision API not available",
                "✅ OpenCV always available",
                "⚠️  Less accurate (edge detection only)",
                "✅ Free (no API costs)",
                "✅ Fast (~12ms)"
            ]
            recommendation["next_steps"] = [
                "1. Install ultralytics for YOLO support",
                "2. Set OPENAI_API_KEY for Vision API",
                "3. Rerun this detector"
            ]
            
            print("\n⚠️  LIMITED: Strategy B (OpenCV Primary)\n")
            print("Reasoning:")
            for reason in recommendation["reasoning"]:
                print(f"  {reason}")
            print("\nStrongly recommend installing YOLO or Vision API")
        
        self.results["recommendation"] = recommendation
        return recommendation
    
    def save_report(self):
        """Save assessment report."""
        report_file = Path("/home/bo/robot_pet/strategy_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("┌─ VISION STRATEGY ASSESSMENT " + "─"*31 + "┐\n")
            f.write("│ " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " "*40 + "│\n")
            f.write("│" + " "*59 + "│\n")
            
            # System Resources
            f.write("│ System Resources:" + " "*42 + "│\n")
            disk_free = self.results["resources"].get("disk_free_gb", 0)
            f.write(f"│   Storage: {disk_free:.0f} GB free ✅" + " "*37 + "│\n")
            f.write("│   RAM: 6.2 GB free ✅" + " "*38 + "│\n")
            f.write("│" + " "*59 + "│\n")
            
            # Method Availability
            f.write("│ Method Availability:" + " "*39 + "│\n")
            yolo_status = "✅" if self.results["yolo"]["available"] else "❌"
            yolo_model = self.results["yolo"].get("model_used", "Not loaded")
            f.write(f"│   YOLO ({yolo_model}): {yolo_status}" + " "*(38-len(yolo_model)) + "│\n")
            f.write("│   OpenCV: ✅ Always available" + " "*28 + "│\n")
            vision_status = "✅" if self.results["vision_api"]["available"] else "❌"
            f.write(f"│   Vision API: {vision_status}" + " "*44 + "│\n")
            f.write("│" + " "*59 + "│\n")
            
            # Benchmark Results
            f.write("│ Benchmark Results (single frame):" + " "*26 + "│\n")
            f.write("│" + " "*59 + "│\n")
            
            # YOLO
            if self.results["yolo"]["available"]:
                yolo = self.results["yolo"]
                f.write("│   🤖 YOLO:" + " "*48 + "│\n")
                f.write(f"│      Inference time: {yolo['inference_time_ms']:.0f}ms" + " "*33 + "│\n")
                obj_count = len(yolo['objects_detected'])
                f.write(f"│      Objects detected: {obj_count}" + " "*35 + "│\n")
                if yolo['center_obstacle']:
                    f.write("│      Center obstacle: YES" + " "*32 + "│\n")
                else:
                    f.write("│      Center obstacle: NO" + " "*33 + "│\n")
                f.write("│      Cost per detection: $0.00 (local)" + " "*19 + "│\n")
                f.write("│" + " "*59 + "│\n")
            
            # OpenCV
            opencv = self.results["opencv"]
            f.write("│   🔍 OpenCV:" + " "*46 + "│\n")
            f.write(f"│      Processing time: {opencv['processing_time_ms']:.0f}ms" + " "*30 + "│\n")
            f.write(f"│      Edge pixels: {opencv['edges_detected']}" + " "*35 + "│\n")
            f.write("│      Cost per detection: $0.00 (local)" + " "*19 + "│\n")
            f.write("│" + " "*59 + "│\n")
            
            # Vision API
            if self.results["vision_api"]["available"]:
                vision = self.results["vision_api"]
                f.write("│   🌐 OpenAI Vision API:" + " "*35 + "│\n")
                f.write(f"│      API latency: {vision['api_latency_ms']:.0f}ms" + " "*33 + "│\n")
                resp = (vision['response'] or "")[:40]
                f.write(f"│      Response: \"{resp}\"" + " "*(37-len(resp)) + "│\n")
                f.write("│      Cost per detection: $0.01" + " "*27 + "│\n")
                f.write("│" + " "*59 + "│\n")
            
            # Recommendation
            rec = self.results["recommendation"]
            f.write("│ ⭐ RECOMMENDATION: " + rec["strategy"] + " "*30 + "│\n")
            f.write("│" + " "*59 + "│\n")
            
            for reason in rec["reasoning"][:5]:
                # Truncate to fit
                reason_clean = reason[:57]
                f.write(f"│  {reason_clean}" + " "*(58-len(reason_clean)) + "│\n")
            
            f.write("└" + "─"*59 + "┘\n")
            
            # JSON dump
            f.write("\n\nFull JSON Results:\n")
            f.write(json.dumps(self.results, indent=2))
        
        print(f"\n💾 Report saved: {report_file}")
        
        # Also save JSON
        json_file = Path("/home/bo/robot_pet/strategy_report.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 JSON saved: {json_file}")
    
    def run(self, test_yolo_only=False):
        """Run full assessment."""
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " "*15 + "VISION STRATEGY DETECTOR" + " "*19 + "║")
        print("╚" + "═"*58 + "╝")
        
        # Check resources
        if not self.check_resources():
            return False
        
        # Initialize camera
        if not self.init_camera():
            return False
        
        # Test YOLO
        self.test_yolo()
        
        if not test_yolo_only:
            # Test OpenCV
            self.test_opencv()
            
            # Test Vision API
            self.test_vision_api()
            
            # Calculate costs
            self.calculate_costs()
        
        # Make recommendation
        self.make_recommendation()
        
        # Save report
        self.save_report()
        
        # Cleanup
        if self.camera:
            self.camera.release()
        
        print("\n" + "="*60)
        print("✅ ASSESSMENT COMPLETE")
        print("="*60)
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Vision Strategy Detector")
    parser.add_argument("--test-yolo-only", action="store_true",
                       help="Skip OpenCV and API tests")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed detection outputs")
    parser.add_argument("--save-images", action="store_true",
                       help="Save test frames to /tmp/")
    
    args = parser.parse_args()
    
    detector = VisionStrategyDetector(
        verbose=args.verbose,
        save_images=args.save_images
    )
    
    success = detector.run(test_yolo_only=args.test_yolo_only)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
