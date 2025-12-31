#!/usr/bin/env python3
"""
Voice Listener Module (Vosk STT)
Local, fast speech recognition for robot pet wake word and commands.

Features:
- Wake word detection ("hey robot", "robot")
- Continuous listening with low latency
- Command extraction after wake word
- Local processing (no API costs, no latency)
"""

import os
import sys
import json
import time
import signal
import queue
import threading
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
    from vosk import Model, KaldiRecognizer
except ImportError:
    print("❌ vosk not installed: pip install vosk")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1  # Mono for Vosk
DTYPE = 'int16'
BLOCK_SIZE = 8000  # ~500ms chunks (larger = less overflow)

# Voice detection
SILENCE_THRESHOLD = 500  # RMS threshold for silence
SPEECH_THRESHOLD = 800   # RMS threshold for speech
MIN_SPEECH_DURATION = 0.3  # Minimum seconds of speech
MAX_COMMAND_DURATION = 8.0  # Maximum seconds to listen for command
SILENCE_AFTER_SPEECH = 1.5  # Silence duration to end command
CONVERSATION_TIMEOUT = 60.0  # Stay awake for 1 minute after wake word

# Wake words (lowercase, Vosk normalizes to lowercase)
WAKE_WORDS = ["hey robot", "robot", "hey bot", "okay robot"]

# Vosk model path
VOSK_MODEL_PATH = os.path.expanduser("~/ml_models/vosk-model-small-en-us-0.15")

# USB Microphone (card 2 based on system audit)
# Will auto-detect if not specified
MIC_DEVICE = None  # None = auto-detect


@dataclass
class ListenerStats:
    """Track listener statistics."""
    wake_words_detected: int = 0
    commands_transcribed: int = 0
    total_audio_seconds: float = 0
    errors: int = 0


stats = ListenerStats()


def find_usb_microphone() -> Optional[int]:
    """Find USB microphone device index."""
    devices = sd.query_devices()
    
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        # Look for USB audio input device
        if dev['max_input_channels'] > 0:
            if 'usb' in name or 'pcm2902' in name or 'pnp' in name:
                print(f"[Voice] 🎤 Found USB mic: {dev['name']} (device {i})")
                return i
    
    # Fallback to default input
    default = sd.default.device[0]
    if default is not None:
        print(f"[Voice] 🎤 Using default input: {devices[default]['name']}")
        return default
    
    return None


class VoiceListener:
    """
    Voice listener with wake word detection.
    
    Usage:
        listener = VoiceListener()
        listener.start()
        
        while True:
            command = listener.get_command()
            if command:
                print(f"Command: {command}")
    """
    
    def __init__(self, model_path: str = VOSK_MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        
        self.running = False
        self.listening_for_command = False
        self.command_queue: queue.Queue = queue.Queue()

        self.mic_device = MIC_DEVICE or find_usb_microphone()
        self.stream: Optional[sd.InputStream] = None

        self._audio_buffer = []
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._conversation_wake_time: Optional[float] = None  # Track when conversation started
        
        self._on_wake_callback: Optional[Callable] = None
        self._on_command_callback: Optional[Callable[[str], None]] = None
        
        self._init_vosk()
    
    def _init_vosk(self):
        """Initialize Vosk model."""
        if not Path(self.model_path).exists():
            print(f"[Voice] ❌ Vosk model not found: {self.model_path}")
            print("        Download from: https://alphacephei.com/vosk/models")
            raise FileNotFoundError(f"Vosk model not found: {self.model_path}")
        
        print(f"[Voice] 🔧 Loading Vosk model...")
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)
        print(f"[Voice] ✅ Vosk ready (offline, fast!)")
    
    def set_wake_callback(self, callback: Callable):
        """Set callback for wake word detection."""
        self._on_wake_callback = callback
    
    def set_command_callback(self, callback: Callable[[str], None]):
        """Set callback for command transcription."""
        self._on_command_callback = callback
    
    def _calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS (volume level) of audio."""
        return np.sqrt(np.mean(audio_data.astype(float) ** 2))
    
    def _process_audio(self, indata: np.ndarray, frames: int, 
                       time_info: dict, status: sd.CallbackFlags):
        """Audio callback - processes incoming audio chunks."""
        if status:
            print(f"[Voice] ⚠️  Audio status: {status}")
        
        # Convert to mono int16 if needed
        audio = indata.copy()
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Take first channel
        
        audio_int16 = (audio * 32767).astype(np.int16) if audio.dtype != np.int16 else audio
        
        rms = self._calculate_rms(audio_int16)
        now = time.time()
        
        # Feed to Vosk
        if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip().lower()
            
            if text:
                self._handle_transcription(text)
        else:
            # Partial result (can use for faster feedback)
            partial = json.loads(self.recognizer.PartialResult())
            partial_text = partial.get("partial", "").strip().lower()
            
            # Check for wake word in partial
            if not self.listening_for_command and partial_text:
                for wake in WAKE_WORDS:
                    if wake in partial_text:
                        self._trigger_wake_word()
                        break
        
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
        
        stats.total_audio_seconds += frames / SAMPLE_RATE
    
    def _trigger_wake_word(self):
        """Handle wake word detection."""
        if self.listening_for_command:
            return  # Already listening

        print(f"\n[Voice] 🔔 Wake word detected!")
        stats.wake_words_detected += 1

        self.listening_for_command = True
        self._conversation_wake_time = time.time()  # Start conversation timer
        self._audio_buffer = []
        self._speech_start_time = None
        self._last_speech_time = None

        # Reset recognizer for clean command capture
        self.recognizer.Reset()

        if self._on_wake_callback:
            self._on_wake_callback()
    
    def _handle_transcription(self, text: str):
        """Handle a full transcription result."""
        if not text:
            return
        
        # Check for wake word
        for wake in WAKE_WORDS:
            if wake in text:
                # Extract command after wake word
                parts = text.split(wake, 1)
                if len(parts) > 1:
                    command = parts[1].strip()
                    if command:
                        self._deliver_command(command)
                        return
                
                # Wake word only - start listening
                self._trigger_wake_word()
                return
        
        # If listening for command, this is the command
        if self.listening_for_command:
            self._deliver_command(text)
    
    def _finalize_command(self):
        """Finalize command listening and get result."""
        if not self.listening_for_command:
            return

        # Get final result
        final = json.loads(self.recognizer.FinalResult())
        text = final.get("text", "").strip()

        # Remove any wake words from the command
        for wake in WAKE_WORDS:
            if text.lower().startswith(wake):
                text = text[len(wake):].strip()

        if text:
            self._deliver_command(text)
        else:
            print("[Voice] ⚠️  No command detected")

        # Reset speech timers but stay in listening mode (conversation timeout will handle exit)
        self._audio_buffer = []
        self._speech_start_time = None
        self._last_speech_time = None
        self._conversation_wake_time = time.time()  # Reset conversation timer
    
    def _deliver_command(self, command: str):
        """Deliver a transcribed command."""
        print(f"[Voice] 📝 Command: \"{command}\"")
        stats.commands_transcribed += 1

        self.command_queue.put(command)
        # Stay in listening mode - reset conversation timer
        self._conversation_wake_time = time.time()

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
    
    def get_stats(self) -> dict:
        """Get listener statistics."""
        return {
            "wake_words_detected": stats.wake_words_detected,
            "commands_transcribed": stats.commands_transcribed,
            "total_audio_seconds": round(stats.total_audio_seconds, 1),
            "errors": stats.errors,
            "is_running": self.running,
            "is_listening_for_command": self.listening_for_command
        }


# Singleton instance
_listener: Optional[VoiceListener] = None


def get_listener() -> VoiceListener:
    """Get or create the global listener instance."""
    global _listener
    if _listener is None:
        _listener = VoiceListener()
    return _listener


# =============================================================================
# CLI Testing
# =============================================================================

def main():
    """Test the voice listener."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Listener (Vosk STT)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--device", "-d", type=int, help="Audio device index")
    parser.add_argument("--test-mic", action="store_true", help="Test microphone levels")
    
    args = parser.parse_args()
    
    if args.list_devices:
        print("\n🎤 Audio Devices:")
        print(sd.query_devices())
        return
    
    if args.test_mic:
        print("\n🎤 Testing microphone levels...")
        print("Speak to see volume levels. Ctrl+C to stop.\n")
        
        device = args.device or find_usb_microphone()
        
        def audio_callback(indata, frames, time_info, status):
            audio = indata[:, 0] if len(indata.shape) > 1 else indata
            rms = np.sqrt(np.mean(audio.astype(float) ** 2))
            bars = int(rms / 100)
            print(f"\rVolume: {'█' * min(bars, 50):<50} {rms:>6.0f}", end="", flush=True)
        
        try:
            with sd.InputStream(device=device, channels=1, samplerate=16000, 
                              dtype='int16', callback=audio_callback):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n✅ Test complete")
        return
    
    # Main listener mode
    print("\n" + "="*60)
    print("🤖 Voice Listener - Robot Pet")
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


