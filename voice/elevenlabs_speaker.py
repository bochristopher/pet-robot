#!/usr/bin/env python3
"""
ElevenLabs TTS Speaker Module
Premium text-to-speech for the robot pet using ElevenLabs API.

Features:
- Async streaming for low latency
- Audio caching to reduce API costs
- Emotion/style control
- Fallback to pyttsx3 if API fails
- Direct playback to USB speaker
"""

import os
import sys
import hashlib
import asyncio
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Cost tracking
api_calls = 0
cached_hits = 0
total_characters = 0

# Cache directory
CACHE_DIR = Path("/tmp/elevenlabs_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Audio device (USB Speaker - card 0)
AUDIO_DEVICE = "plughw:0,0"

# ElevenLabs voice presets for robot pet personality
# Using one consistent voice (Rachel) with different settings for emotions
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel - consistent voice across all emotions

VOICE_PRESETS = {
    "friendly": {
        "voice_id": VOICE_ID,
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
    },
    "curious": {
        "voice_id": VOICE_ID,
        "stability": 0.4,
        "similarity_boost": 0.8,
        "style": 0.3,
    },
    "excited": {
        "voice_id": VOICE_ID,
        "stability": 0.3,
        "similarity_boost": 0.75,
        "style": 0.5,
    },
    "calm": {
        "voice_id": VOICE_ID,
        "stability": 0.7,
        "similarity_boost": 0.6,
        "style": 0.0,
    },
    "playful": {
        "voice_id": VOICE_ID,
        "stability": 0.35,
        "similarity_boost": 0.75,
        "style": 0.4,
    },
    "apologetic": {
        "voice_id": VOICE_ID,
        "stability": 0.6,
        "similarity_boost": 0.7,
        "style": 0.1,
    },
}

# Default voice
DEFAULT_VOICE = "friendly"


@dataclass
class SpeakerStats:
    """Track speaker usage statistics."""
    api_calls: int = 0
    cached_hits: int = 0
    total_characters: int = 0
    fallback_uses: int = 0


stats = SpeakerStats()


def get_cache_path(text: str, voice_id: str) -> Path:
    """Generate cache file path based on text and voice."""
    key = f"{voice_id}:{text}"
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{hash_key}.mp3"


def check_cache(text: str, voice_id: str) -> Optional[Path]:
    """Check if audio is cached."""
    cache_path = get_cache_path(text, voice_id)
    if cache_path.exists():
        return cache_path
    return None


def play_audio_file(audio_path: Path, blocking: bool = True) -> bool:
    """Play an audio file through USB speaker."""
    audio_path = Path(audio_path)

    # Try GStreamer first (handles mp3 natively)
    try:
        cmd = [
            "gst-launch-1.0", "-q",
            "filesrc", f"location={audio_path}",
            "!", "decodebin",
            "!", "audioconvert",
            "!", "volume", "volume=0.4",  # Reduce volume to 40%
            "!", "audioresample",
            "!", "alsasink", f"device={AUDIO_DEVICE}"
        ]

        if blocking:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True

    except Exception as e:
        print(f"[Speaker] GStreamer error: {e}")

    # Fallback to paplay (PulseAudio)
    try:
        cmd = ["paplay", str(audio_path)]
        if blocking:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except Exception as e:
        print(f"[Speaker] paplay error: {e}")

    # Last resort: try aplay with wav files
    try:
        if audio_path.suffix == ".wav":
            cmd = ["aplay", "-D", AUDIO_DEVICE, str(audio_path)]
            if blocking:
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                return result.returncode == 0
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
    except Exception as e:
        print(f"[Speaker] aplay error: {e}")

    print(f"[Speaker] Play error: No working audio player found")
    return False


def speak_pyttsx3_fallback(text: str) -> bool:
    """Fallback to pyttsx3 for offline TTS."""
    global stats
    stats.fallback_uses += 1
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        # Save to temp file and play through USB speaker
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        
        # Play through USB speaker
        subprocess.run(["aplay", "-D", AUDIO_DEVICE, temp_path], 
                      capture_output=True, timeout=30)
        Path(temp_path).unlink(missing_ok=True)
        
        print(f"[Speaker] Fallback TTS: {text[:50]}...")
        return True
        
    except Exception as e:
        print(f"[Speaker] Fallback failed: {e}")
        # Last resort: espeak
        try:
            subprocess.run(["espeak", "-s", "150", text], timeout=30)
            return True
        except:
            return False


class ElevenLabsSpeaker:
    """
    ElevenLabs TTS with caching and emotion control.
    
    Usage:
        speaker = ElevenLabsSpeaker()
        speaker.speak("Hello world!", emotion="excited")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.client = None
        self.available = False
        
        if not self.api_key:
            print("[Speaker] ⚠️  ELEVENLABS_API_KEY not set - will use fallback TTS")
        else:
            self._init_client()
    
    def _init_client(self):
        """Initialize ElevenLabs client."""
        try:
            from elevenlabs.client import ElevenLabs

            self.client = ElevenLabs(api_key=self.api_key)
            self.available = True
            print(f"[Speaker] ✅ ElevenLabs initialized (key: {self.api_key[:8]}...)")

        except ImportError:
            print("[Speaker] ⚠️  elevenlabs library not installed")
            print("         Run: pip install elevenlabs")
            self.available = False
        except Exception as e:
            print(f"[Speaker] ⚠️  ElevenLabs init failed: {e}")
            self.available = False
    
    def get_voice_settings(self, emotion: str = DEFAULT_VOICE) -> dict:
        """Get voice settings for an emotion preset."""
        return VOICE_PRESETS.get(emotion, VOICE_PRESETS[DEFAULT_VOICE])
    
    def speak(self, text: str, emotion: str = DEFAULT_VOICE, blocking: bool = True) -> bool:
        """
        Speak text using ElevenLabs TTS.
        
        Args:
            text: Text to speak
            emotion: Emotion preset (friendly, curious, excited, calm, playful)
            blocking: Wait for playback to complete
            
        Returns:
            True if successful
        """
        global stats
        
        if not text or not text.strip():
            return False
        
        text = text.strip()
        preset = self.get_voice_settings(emotion)
        voice_id = preset["voice_id"]
        
        # Check cache first
        cached = check_cache(text, voice_id)
        if cached:
            stats.cached_hits += 1
            print(f"[Speaker] 📀 Cache hit: {text[:40]}...")
            return play_audio_file(cached, blocking=blocking)
        
        # Generate with ElevenLabs
        if not self.available:
            return speak_pyttsx3_fallback(text)
        
        try:
            stats.api_calls += 1
            stats.total_characters += len(text)

            print(f"[Speaker] 🎙️  Generating ({emotion}): {text[:50]}...")

            # Generate audio using v2.x API
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                voice_settings={
                    "stability": preset["stability"],
                    "similarity_boost": preset["similarity_boost"],
                    "style": preset.get("style", 0.0),
                    "use_speaker_boost": True
                }
            )

            # Save to cache
            cache_path = get_cache_path(text, voice_id)
            with open(cache_path, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)

            print(f"[Speaker] 💾 Cached: {cache_path.name}")

            # Play audio
            result = play_audio_file(cache_path, blocking=blocking)

            # Small delay to prevent cutoff and let audio fully finish
            if blocking:
                time.sleep(0.3)

            return result

        except Exception as e:
            print(f"[Speaker] ❌ ElevenLabs error: {e}")
            return speak_pyttsx3_fallback(text)
    
    def speak_async(self, text: str, emotion: str = DEFAULT_VOICE):
        """Non-blocking speak (runs in background thread)."""
        thread = threading.Thread(
            target=self.speak, 
            args=(text, emotion, True),
            daemon=True
        )
        thread.start()
        return thread
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "api_calls": stats.api_calls,
            "cached_hits": stats.cached_hits,
            "total_characters": stats.total_characters,
            "fallback_uses": stats.fallback_uses,
            "estimated_cost": stats.total_characters * 0.00003,  # ~$0.03 per 1000 chars
            "cache_files": len(list(CACHE_DIR.glob("*.mp3")))
        }
    
    def clear_cache(self):
        """Clear audio cache."""
        for f in CACHE_DIR.glob("*.mp3"):
            f.unlink()
        print("[Speaker] Cache cleared")
    
    def list_voices(self) -> list:
        """List available voices from ElevenLabs."""
        if not self.available:
            return []
        
        try:
            voices = self.client.voices.get_all()
            return [{"id": v.voice_id, "name": v.name} for v in voices.voices]
        except Exception as e:
            print(f"[Speaker] Error listing voices: {e}")
            return []


# Singleton instance
_speaker: Optional[ElevenLabsSpeaker] = None


def get_speaker() -> ElevenLabsSpeaker:
    """Get or create the global speaker instance."""
    global _speaker
    if _speaker is None:
        _speaker = ElevenLabsSpeaker()
    return _speaker


def speak(text: str, emotion: str = DEFAULT_VOICE, blocking: bool = True) -> bool:
    """Convenience function for speaking."""
    return get_speaker().speak(text, emotion, blocking)


# ============================================================================
# CLI Testing
# ============================================================================

def main():
    """Test the speaker with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ElevenLabs TTS Speaker")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--emotion", "-e", default="friendly", 
                       choices=list(VOICE_PRESETS.keys()),
                       help="Emotion preset")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--stats", action="store_true", help="Show usage stats")
    parser.add_argument("--clear-cache", action="store_true", help="Clear audio cache")
    parser.add_argument("--test-all", action="store_true", help="Test all emotion presets")
    
    args = parser.parse_args()
    
    speaker = get_speaker()
    
    if args.list_voices:
        print("\n🎤 Available ElevenLabs Voices:")
        voices = speaker.list_voices()
        for v in voices[:20]:
            print(f"  {v['id']}: {v['name']}")
        return
    
    if args.stats:
        print("\n📊 Speaker Statistics:")
        for k, v in speaker.get_stats().items():
            print(f"  {k}: {v}")
        return
    
    if args.clear_cache:
        speaker.clear_cache()
        return
    
    if args.test_all:
        print("\n🧪 Testing all emotion presets...")
        for emotion in VOICE_PRESETS.keys():
            print(f"\n--- {emotion.upper()} ---")
            speaker.speak(f"I am feeling {emotion} right now!", emotion=emotion)
            time.sleep(0.5)
        return
    
    if args.text:
        print(f"\n🔊 Speaking with {args.emotion} emotion...")
        success = speaker.speak(args.text, emotion=args.emotion)
        print(f"{'✅ Success' if success else '❌ Failed'}")
        print(f"\n📊 Stats: {speaker.get_stats()}")
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("🤖 ElevenLabs TTS Speaker - Interactive Mode")
        print("="*60)
        print(f"API Key: {'✅ Set' if speaker.available else '❌ Not set'}")
        print(f"Cache: {CACHE_DIR}")
        print(f"Emotions: {', '.join(VOICE_PRESETS.keys())}")
        print("\nType text to speak (prefix with @emotion to change)")
        print("Example: @excited Hello world!")
        print("Press Ctrl+C to exit\n")
        
        try:
            while True:
                user_input = input("🎤 > ").strip()
                if not user_input:
                    continue
                
                # Parse emotion prefix
                emotion = "friendly"
                text = user_input
                
                if user_input.startswith("@"):
                    parts = user_input.split(" ", 1)
                    if len(parts) == 2:
                        emotion_str = parts[0][1:]
                        if emotion_str in VOICE_PRESETS:
                            emotion = emotion_str
                            text = parts[1]
                
                speaker.speak(text, emotion=emotion)
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            print(f"📊 Final stats: {speaker.get_stats()}")


if __name__ == "__main__":
    main()


