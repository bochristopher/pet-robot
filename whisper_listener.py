#!/usr/bin/env python3
"""
Voice Listener Module (OpenAI Whisper API)
Cloud-based, accurate speech recognition for robot pet wake word and commands.

Features:
- Wake word detection ("hey robot", "robot")
- Continuous listening with accurate transcription
- Command extraction after wake word
- Uses OpenAI Whisper API for high accuracy
"""

import os
import sys
import time
import wave
import queue
import threading
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

# Try to import required libraries
try:
    import sounddevice as sd
except ImportError:
    print("❌ sounddevice not installed: pip install sounddevice")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("❌ openai not installed: pip install openai")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1  # Mono
DTYPE = 'int16'
BLOCK_SIZE = 16000  # ~1000ms chunks (increased from 8000 to reduce overflow)

# Voice detection
SILENCE_THRESHOLD = 400  # RMS threshold for silence (lowered from 500)
SPEECH_THRESHOLD = 600   # RMS threshold for speech (lowered from 800)
MIN_SPEECH_DURATION = 0.3  # Minimum seconds of speech (lowered from 0.5)
MAX_COMMAND_DURATION = 8.0  # Maximum seconds to listen for command
SILENCE_AFTER_SPEECH = 1.2  # Silence duration to end command (lowered from 1.5)
CONVERSATION_TIMEOUT = 60.0  # Stay awake for 1 minute after wake word

# Wake words (lowercase) - includes common misrecognitions
WAKE_WORDS = [
    "hey robot", "robot", "hey bot", "okay robot", "ok robot",
    "a robot", "hey robotic", "hey robert", "air robot",
    "hey trouble", "hey robux"  # Common Whisper mishears
]

# USB Microphone
MIC_DEVICE = None  # None = auto-detect

# Whisper API costs
WHISPER_COST_PER_MINUTE = 0.006  # $0.006 per minute


@dataclass
class ListenerStats:
    """Track listener statistics."""
    wake_words_detected: int = 0
    commands_transcribed: int = 0
    total_audio_seconds: float = 0
    api_calls: int = 0
    errors: int = 0

    def get_cost_estimate(self) -> float:
        """Estimate API cost."""
        return (self.total_audio_seconds / 60) * WHISPER_COST_PER_MINUTE


stats = ListenerStats()


def find_usb_microphone() -> Optional[int]:
    """Find USB microphone device index."""
    devices = sd.query_devices()

    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        # Look for USB audio input device
        if dev['max_input_channels'] > 0:
            if 'usb' in name or 'pcm2902' in name or 'pnp' in name or 'hd 1080p' in name:
                print(f"[Voice] 🎤 Found USB mic: {dev['name']} (device {i})")
                return i

    # Fallback to default input
    default = sd.default.device[0]
    if default is not None:
        print(f"[Voice] 🎤 Using default input: {devices[default]['name']}")
        return default

    return None


class WhisperListener:
    """
    Voice listener with wake word detection using OpenAI Whisper API.

    Usage:
        listener = WhisperListener()
        listener.start()

        while True:
            command = listener.get_command()
            if command:
                print(f"Command: {command}")
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=self.api_key)

        self.running = False
        self.listening_for_command = False
        self.paused = False  # Pause during speech output
        self.command_queue: queue.Queue = queue.Queue()

        self.mic_device = MIC_DEVICE or find_usb_microphone()
        self.stream: Optional[sd.InputStream] = None

        self._audio_buffer = []
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._conversation_wake_time: Optional[float] = None

        self._on_wake_callback: Optional[Callable] = None
        self._on_command_callback: Optional[Callable[[str], None]] = None

        print(f"[Voice] ✅ Whisper API ready (key: {self.api_key[:8]}...)")

    def set_wake_callback(self, callback: Callable):
        """Set callback for wake word detection."""
        self._on_wake_callback = callback

    def set_command_callback(self, callback: Callable[[str], None]):
        """Set callback for command transcription."""
        self._on_command_callback = callback

    def _calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS (volume level) of audio."""
        return np.sqrt(np.mean(audio_data.astype(float) ** 2))

    def _transcribe_audio(self, audio_buffer: list) -> Optional[str]:
        """Transcribe audio buffer using Whisper API."""
        if not audio_buffer:
            return None

        try:
            # Concatenate audio chunks
            audio_data = np.concatenate(audio_buffer)

            # Save to temporary WAV file
            temp_path = tempfile.mktemp(suffix='.wav')
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data.tobytes())

            # Transcribe with Whisper
            stats.api_calls += 1
            duration = len(audio_data) / SAMPLE_RATE
            stats.total_audio_seconds += duration

            print(f"[Voice] 🌐 Transcribing {duration:.1f}s audio...")

            with open(temp_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )

            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass

            text = transcript.text.strip() if transcript else ""
            return text if text else None

        except Exception as e:
            stats.errors += 1
            print(f"[Voice] ❌ Transcription error: {e}")
            return None

    def _process_audio(self, indata: np.ndarray, frames: int,
                       time_info: dict, status: sd.CallbackFlags):
        """Audio callback - processes incoming audio chunks."""
        # Ignore audio if paused (robot is speaking)
        if self.paused:
            return

        if status:
            print(f"[Voice] ⚠️  Audio status: {status}")

        # Convert to mono int16 if needed
        audio = indata.copy()
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Take first channel

        audio_int16 = (audio * 32767).astype(np.int16) if audio.dtype != np.int16 else audio

        rms = self._calculate_rms(audio_int16)
        now = time.time()

        # Track speech for command listening
        if self.listening_for_command:
            # Check conversation timeout (1 minute of inactivity)
            if self._conversation_wake_time and (now - self._conversation_wake_time) > CONVERSATION_TIMEOUT:
                print("[Voice] 💤 Conversation timeout - going back to sleep")
                self.listening_for_command = False
                self._conversation_wake_time = None
                self._audio_buffer = []
                self._speech_start_time = None
                self._last_speech_time = None
                return

            if rms > SPEECH_THRESHOLD:
                if self._speech_start_time is None:
                    self._speech_start_time = now
                self._last_speech_time = now
                self._audio_buffer.append(audio_int16.copy())
            elif rms < SILENCE_THRESHOLD:
                if self._last_speech_time and (now - self._last_speech_time) > SILENCE_AFTER_SPEECH:
                    # Silence after speech - finalize command
                    self._finalize_command()

            # Timeout check for single command
            if self._speech_start_time and (now - self._speech_start_time) > MAX_COMMAND_DURATION:
                self._finalize_command()
        else:
            # Always buffer audio to detect wake word
            if rms > SPEECH_THRESHOLD:
                if self._speech_start_time is None:
                    self._speech_start_time = now
                self._last_speech_time = now
                self._audio_buffer.append(audio_int16.copy())
            elif rms < SILENCE_THRESHOLD:
                if self._last_speech_time and (now - self._last_speech_time) > SILENCE_AFTER_SPEECH:
                    # Check for wake word
                    if self._audio_buffer and (now - self._speech_start_time) > MIN_SPEECH_DURATION:
                        self._check_wake_word()

    def _check_wake_word(self):
        """Check if audio contains wake word."""
        text = self._transcribe_audio(self._audio_buffer)

        self._audio_buffer = []
        self._speech_start_time = None
        self._last_speech_time = None

        if not text:
            return

        text_lower = text.lower()
        print(f"[Voice] 📝 Heard: \"{text}\"")

        # Check for wake word
        for wake in WAKE_WORDS:
            if wake in text_lower:
                # Extract command after wake word
                parts = text_lower.split(wake, 1)
                if len(parts) > 1:
                    command = parts[1].strip()
                    if command:
                        self._trigger_wake_word()
                        self._deliver_command(command)
                        return

                # Wake word only - start listening
                self._trigger_wake_word()
                return

        # No wake word detected - give user feedback
        print(f"[Voice] 💤 No wake word detected. Say 'hey robot' to activate.")

    def _trigger_wake_word(self):
        """Handle wake word detection."""
        if self.listening_for_command:
            return  # Already listening

        print(f"\n[Voice] 🔔 Wake word detected!")
        stats.wake_words_detected += 1

        self.listening_for_command = True
        self._conversation_wake_time = time.time()
        self._audio_buffer = []
        self._speech_start_time = None
        self._last_speech_time = None

        if self._on_wake_callback:
            self._on_wake_callback()

    def _finalize_command(self):
        """Finalize command listening and get result."""
        if not self.listening_for_command:
            return

        text = self._transcribe_audio(self._audio_buffer)

        # Reset buffers but stay in listening mode
        self._audio_buffer = []
        self._speech_start_time = None
        self._last_speech_time = None
        self._conversation_wake_time = time.time()  # Reset conversation timer

        if text:
            # Remove any wake words from the command
            text_lower = text.lower()
            for wake in WAKE_WORDS:
                if text_lower.startswith(wake):
                    text = text[len(wake):].strip()
                    break

            if text:
                self._deliver_command(text)
        else:
            print("[Voice] ⚠️  No command detected")

    def _deliver_command(self, command: str):
        """Deliver a transcribed command."""
        print(f"[Voice] 📝 Command: \"{command}\"")
        stats.commands_transcribed += 1

        self.command_queue.put(command)
        self._conversation_wake_time = time.time()  # Reset conversation timer

        if self._on_command_callback:
            self._on_command_callback(command)

    def start(self):
        """Start listening."""
        if self.running:
            return

        print(f"\n[Voice] 🎤 Starting listener...")
        print(f"[Voice] Wake words: {WAKE_WORDS}")

        self.running = True

        self.stream = sd.InputStream(
            device=self.mic_device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._process_audio
        )
        self.stream.start()

        print(f"[Voice] ✅ Listening! Say '{WAKE_WORDS[0]}' to activate")

    def stop(self):
        """Stop listening."""
        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        print("[Voice] 🛑 Listener stopped")

    def get_command(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Get the next command (blocking).

        Args:
            timeout: Seconds to wait (None = forever)

        Returns:
            Command string or None if timeout
        """
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_command_nowait(self) -> Optional[str]:
        """Get command without blocking (returns None if no command)."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def is_listening(self) -> bool:
        """Check if currently listening for a command."""
        return self.listening_for_command

    def trigger_listen(self):
        """Manually trigger command listening (skip wake word)."""
        self._trigger_wake_word()

    def pause(self):
        """Pause listening (e.g., during robot speech)."""
        self.paused = True
        # Clear buffers to avoid hearing tail end of speech
        self._audio_buffer = []
        self._speech_start_time = None
        self._last_speech_time = None

    def resume(self):
        """Resume listening after pause."""
        self.paused = False

    def get_stats(self) -> dict:
        """Get listener statistics."""
        return {
            "wake_words_detected": stats.wake_words_detected,
            "commands_transcribed": stats.commands_transcribed,
            "api_calls": stats.api_calls,
            "total_audio_seconds": round(stats.total_audio_seconds, 1),
            "estimated_cost": f"${stats.get_cost_estimate():.4f}",
            "errors": stats.errors,
            "is_running": self.running,
            "is_listening_for_command": self.listening_for_command
        }


# Singleton instance
_listener: Optional[WhisperListener] = None


def get_listener() -> WhisperListener:
    """Get or create the global listener instance."""
    global _listener
    if _listener is None:
        _listener = WhisperListener()
    return _listener


# =============================================================================
# CLI Testing
# =============================================================================

def main():
    """Test the voice listener."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Voice Listener (Whisper API)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--device", "-d", type=int, help="Audio device index")

    args = parser.parse_args()

    if args.list_devices:
        print("\n🎤 Audio Devices:")
        print(sd.query_devices())
        return

    # Main listener mode
    print("\n" + "="*60)
    print("🤖 Voice Listener - Whisper API")
    print("="*60)

    if args.device:
        global MIC_DEVICE
        MIC_DEVICE = args.device

    listener = get_listener()

    def on_wake():
        print("🔔 ACTIVATED - listening for command...")

    def on_command(cmd):
        print(f"✅ Got command: {cmd}")

    listener.set_wake_callback(on_wake)
    listener.set_command_callback(on_command)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n\n👋 Shutting down...")
        listener.stop()
        print(f"📊 Stats: {listener.get_stats()}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    listener.start()

    print("\n🎤 Say 'hey robot' followed by a command!")
    print("   Example: 'hey robot what do you see'")
    print("   Press Ctrl+C to stop\n")

    # Main loop
    while True:
        command = listener.get_command(timeout=1.0)
        if command:
            print(f"\n💬 Processing: '{command}'")
            print(f"📊 Stats: {listener.get_stats()}\n")


if __name__ == "__main__":
    main()
