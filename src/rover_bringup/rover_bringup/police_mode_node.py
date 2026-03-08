#!/usr/bin/env python3
"""
Police Mode Node (NO MICROPHONE)
================================
Detects people and demands documentation with authority.
Speech output only - no microphone/listening.
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
from std_msgs.msg import Bool, String
import subprocess
import random
import time
import threading
import os
import hashlib
import tempfile

# Try to import requests for ElevenLabs API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class PoliceModeNode(Node):
    """Documentation enforcement robot - speech only, no microphone."""
    
    # ElevenLabs deep male voices
    ELEVENLABS_VOICES = {
        'adam': 'pNInz6obpgDQGcFmaJgB',      # Deep American - DEEPEST
        'daniel': 'onwK4e9ZLuTAKqWW03F9',    # Deep British
        'charlie': 'IKne3meq5aSn9XLyUdCD',   # Deep Australian  
        'james': 'ZQe5CZNOzWyzPSCn5a3c',     # Deep authoritative
        'josh': 'TxGEqnHWrfWFTfGW9XjX',
        'sam': 'yoZ06aMxZJJ28mfd3POQ',
    }
    
    DOCUMENT_CLASSES = [
        'book', 'cell phone', 'laptop', 'remote', 'keyboard',
        'handbag', 'suitcase', 'backpack', 'wallet'
    ]
    
    INITIAL_PHRASES = [
        "HALT! Police check. Show your documentation immediately!",
        "STOP! Police. Papers, please!",
        "You there! Police check. Present your documents!",
        "FREEZE! Police document check in progress!",
        "CITIZEN! Police request: show identification now!",
        "HEY! Police here. Where are your papers?",
    ]
    
    FOLLOWUP_PHRASES = [
        "I am still waiting for your documentation!",
        "Show me your papers now!",
        "Do not attempt to flee! Show your papers!",
        "My patience is running thin. Papers. Now!",
        "I will follow you until you show identification!",
        "This is your final warning! Present documents!",
    ]
    
    DOCUMENT_VERIFIED_PHRASES = [
        "Document detected. Thank you for your cooperation.",
        "Papers verified. You may proceed.",
        "Documentation confirmed. Move along.",
    ]
    
    GOODBYE_PHRASES = [
        "That's right, keep moving! Your status remains unverified!",
        "I will be watching for you!",
        "Suspicious behavior noted!",
        "This isn't over!",
    ]
    
    def __init__(self):
        super().__init__('police_mode_node')
        
        # Parameters
        self.declare_parameter('enabled', True)
        self.declare_parameter('elevenlabs_api_key', '')
        self.declare_parameter('elevenlabs_voice', 'josh')
        self.declare_parameter('use_elevenlabs', True)
        self.declare_parameter('cache_audio', True)
        self.declare_parameter('espeak_speed', 130)
        self.declare_parameter('espeak_pitch', 20)
        self.declare_parameter('initial_cooldown', 8.0)
        self.declare_parameter('followup_cooldown', 6.0)
        self.declare_parameter('tracking_enabled', True)
        self.declare_parameter('tracking_speed', 0.15)
        self.declare_parameter('tracking_angular', 0.8)
        self.declare_parameter('confidence_threshold', 0.5)
        
        self.enabled = self.get_parameter('enabled').value
        self.use_elevenlabs = self.get_parameter('use_elevenlabs').value
        self.cache_audio = self.get_parameter('cache_audio').value
        self.espeak_speed = self.get_parameter('espeak_speed').value
        self.espeak_pitch = self.get_parameter('espeak_pitch').value
        self.initial_cooldown = self.get_parameter('initial_cooldown').value
        self.followup_cooldown = self.get_parameter('followup_cooldown').value
        self.tracking_enabled = self.get_parameter('tracking_enabled').value
        self.tracking_speed = self.get_parameter('tracking_speed').value
        self.tracking_angular = self.get_parameter('tracking_angular').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        
        # ElevenLabs setup
        api_key = self.get_parameter('elevenlabs_api_key').value or os.environ.get('ELEVENLABS_API_KEY', '')
        voice_name = self.get_parameter('elevenlabs_voice').value.lower()
        self.elevenlabs_api_key = api_key
        self.elevenlabs_voice_id = self.ELEVENLABS_VOICES.get(voice_name, self.ELEVENLABS_VOICES['adam'])
        
        self.cache_dir = os.path.join(tempfile.gettempdir(), 'police_mode_cache')
        if self.cache_audio:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.elevenlabs_available = bool(self.use_elevenlabs and REQUESTS_AVAILABLE and self.elevenlabs_api_key)
        
        # State
        self.person_detected = False
        self.document_shown = False
        self.last_person_seen_time = 0.0  # Time we last saw a person
        self.person_lost_grace = 3.0  # Seconds to wait before declaring person "lost"
        self.initial_phrase_lockout = 20.0  # Don't repeat initial phrase for this long
        self.person_first_seen = 0.0
        self.last_speech_time = 0.0
        self.followup_count = 0
        self.is_speaking = False
        self.speech_lock = threading.Lock()
        
        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_callback, 10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/police_mode/status', 10)
        self.alert_pub = self.create_publisher(Bool, '/police_mode/alert', 10)
        self.verified_pub = self.create_publisher(Bool, '/police_mode/document_verified', 10)
        
        # Services
        self.enable_srv = self.create_service(SetBool, '/police_mode/enable', self.enable_callback)
        
        # Timer
        self.create_timer(1.0, self.status_timer_callback)
        
        # Startup
        self.get_logger().info("=" * 50)
        self.get_logger().info("  POLICE MODE - SPEECH ONLY")
        self.get_logger().info(f"  Voice: {'ElevenLabs' if self.elevenlabs_available else 'espeak'}")
        self.get_logger().info(f"  Tracking: {'ON' if self.tracking_enabled else 'OFF'}")
        self.get_logger().info("=" * 50)
        
        if self.enabled:
            self.speak_threaded("Police Mode activated. Scanning for undocumented individuals.")
    
    def _get_cache_path(self, text: str) -> str:
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"{text_hash}.mp3")
    
    def speak_elevenlabs(self, text: str) -> bool:
        if not self.elevenlabs_available:
            return False
        try:
            cache_path = self._get_cache_path(text)
            
            # Play cached audio if exists
            if self.cache_audio and os.path.exists(cache_path):
                self._play_audio(cache_path)
                return True
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}"
            headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": self.elevenlabs_api_key}
            # Deep dramatic voice settings
            data = {
                "text": text, 
                "model_id": "eleven_multilingual_v2", 
                "voice_settings": {
                    "stability": 0.35,
                    "similarity_boost": 0.85,
                    "style": 0.6,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=20)
            if response.status_code == 200:
                audio_path = cache_path if self.cache_audio else os.path.join(tempfile.gettempdir(), 'police_speech.mp3')
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                self._play_audio(audio_path)
                return True
            else:
                self.get_logger().warn(f"ElevenLabs API returned {response.status_code}")
        except Exception as e:
            self.get_logger().warn(f"ElevenLabs error: {e}")
        return False
    
    def _play_audio(self, audio_path: str):
        """Play audio file using available player."""
        try:
            # Use ffplay (most reliable on Jetson)
            subprocess.run(
                ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', audio_path],
                timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            try:
                # Fallback to paplay
                subprocess.run(['paplay', audio_path], timeout=60)
            except Exception:
                pass
    
    def speak_espeak(self, text: str):
        try:
            subprocess.run(['espeak', '-s', str(self.espeak_speed), '-p', str(self.espeak_pitch), '-v', 'en-us+m3', text], timeout=30)
        except Exception as e:
            self.get_logger().debug(f"espeak error: {e}")
    
    def speak(self, text: str):
        try:
            if not self.speak_elevenlabs(text):
                self.speak_espeak(text)
        finally:
            with self.speech_lock:
                self.is_speaking = False
    
    def speak_threaded(self, text: str):
        with self.speech_lock:
            if self.is_speaking:
                return False
            self.is_speaking = True
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()
        self.get_logger().info(f"🔊 {text}")
        return True
    
    def detection_callback(self, msg: Detection2DArray):
        if not self.enabled:
            return
        
        current_time = time.time()
        person_found = False
        document_found = False
        best_person = None
        best_conf = 0.0
        detected_items = []
        
        for det in msg.detections:
            for res in det.results:
                cls = res.hypothesis.class_id.lower()
                conf = res.hypothesis.score
                if cls == 'person' and conf >= self.confidence_threshold:
                    person_found = True
                    if conf > best_conf:
                        best_conf = conf
                        best_person = det
                if cls in self.DOCUMENT_CLASSES and conf >= 0.35:
                    document_found = True
                    detected_items.append(cls)
        
        # Update last seen time if person found
        if person_found:
            self.last_person_seen_time = current_time
        
        # Handle document shown (even if person flickers out briefly)
        if document_found and self.person_detected and not self.document_shown:
            self.get_logger().info(f"📄 Document detected: {detected_items}")
            self.handle_document_shown()
        elif person_found and best_person:
            self.handle_person_detected(best_person, best_conf, current_time)
        elif not person_found:
            # Only handle "lost" if enough time passed since last seen
            time_since_seen = current_time - self.last_person_seen_time
            if time_since_seen > self.person_lost_grace:
                self.handle_person_lost(current_time)
    
    def handle_document_shown(self):
        if not self.document_shown:
            self.document_shown = True
            self.get_logger().info("✅ DOCUMENT VERIFIED")
            self.speak_threaded(random.choice(self.DOCUMENT_VERIFIED_PHRASES))
            if self.tracking_enabled:
                self.cmd_vel_pub.publish(Twist())
            msg = Bool(); msg.data = True; self.verified_pub.publish(msg)
            msg = Bool(); msg.data = False; self.alert_pub.publish(msg)
    
    def handle_person_detected(self, detection, confidence, current_time):
        was_detected = self.person_detected
        self.person_detected = True
        
        # Document already shown for this encounter? Stay quiet
        if self.document_shown:
            return
        
        msg = Bool(); msg.data = True; self.alert_pub.publish(msg)
        
        bbox = detection.bbox
        person_x = bbox.center.position.x
        person_size = bbox.size_x * bbox.size_y
        
        # Determine if this is truly a NEW encounter
        # New = wasn't detected AND enough time since last initial phrase
        time_since_initial = current_time - self.person_first_seen
        is_new_encounter = not was_detected and time_since_initial > self.initial_phrase_lockout
        
        if is_new_encounter:
            # Fresh encounter - reset state and give initial warning
            self.person_first_seen = current_time
            self.followup_count = 0
            self.last_speech_time = current_time
            self.document_shown = False
            self.speak_threaded(random.choice(self.INITIAL_PHRASES))
            self.get_logger().warn(f"🚨 NEW PERSON DETECTED! {confidence:.0%}")
        elif was_detected and (current_time - self.last_speech_time) > self.followup_cooldown:
            # Same person, time for follow-up
            self.followup_count += 1
            self.last_speech_time = current_time
            self.speak_threaded(random.choice(self.FOLLOWUP_PHRASES))
            self.get_logger().info(f"📢 Follow-up #{self.followup_count}")
        elif not was_detected:
            # Person reappeared within lockout - just log, no speech
            self.get_logger().debug(f"Person reappeared (within lockout)")
        
        if self.tracking_enabled:
            twist = Twist()
            offset = (person_x - 320) / 320
            twist.angular.z = -offset * self.tracking_angular
            if person_size < 50000:
                twist.linear.x = self.tracking_speed
            self.cmd_vel_pub.publish(twist)
    
    def handle_person_lost(self, current_time):
        if self.person_detected:
            elapsed = current_time - self.person_first_seen
            if not self.document_shown and elapsed > 3.0:
                self.speak_threaded(random.choice(self.GOODBYE_PHRASES))
                self.get_logger().info("👋 Person left without showing documents")
            elif self.document_shown:
                self.get_logger().info("✅ Person verified and departed")
            self.person_detected = False
            # Keep document_shown = True until next NEW encounter
            # This prevents re-demanding from someone who just showed papers
            if self.tracking_enabled:
                self.cmd_vel_pub.publish(Twist())
            msg = Bool(); msg.data = False; self.alert_pub.publish(msg)
    
    def enable_callback(self, request, response):
        self.enabled = request.data
        if self.enabled:
            self.speak_threaded("Police Mode reactivated. Resuming patrol.")
        else:
            self.speak_threaded("Standing down. Patrol suspended.")
            self.cmd_vel_pub.publish(Twist())
            self.person_detected = False
        response.success = True
        response.message = f"Police Mode {'enabled' if self.enabled else 'disabled'}"
        return response
    
    def status_timer_callback(self):
        msg = String()
        if not self.enabled:
            msg.data = "DISABLED"
        elif self.document_shown:
            msg.data = "VERIFIED"
        elif self.person_detected:
            elapsed = time.time() - self.person_first_seen
            msg.data = f"ALERT: Person {elapsed:.0f}s (follow-ups: {self.followup_count})"
        else:
            msg.data = "PATROLLING"
        self.status_pub.publish(msg)
    
    def destroy_node(self):
        self.cmd_vel_pub.publish(Twist())
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoliceModeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
