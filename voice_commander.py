#!/usr/bin/env python3
"""
Voice Commander - Whisper API based voice control
Listens for commands and controls the robot.
"""

import os
import io
import time
import wave
import queue
import threading
import numpy as np
import sounddevice as sd
from openai import OpenAI

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'

# Voice detection
SILENCE_THRESHOLD = 500
SPEECH_THRESHOLD = 1000
MIN_SPEECH_DURATION = 0.3
MAX_SPEECH_DURATION = 5.0
SILENCE_AFTER_SPEECH = 1.0

# Wake words (checked locally before API call to save costs)
WAKE_WORDS = ["hey robot", "robot", "hey bot", "okay robot", "yo robot"]


class VoiceCommander:
    """Voice command listener using Whisper API."""

    def __init__(self):
        self.client = OpenAI()
        self.running = False
        self.command_queue = queue.Queue()
        self.callback = None

        # Audio state
        self._audio_buffer = []
        self._is_speaking = False
        self._speech_start = None
        self._last_speech = None
        self._stream = None
        self._listen_thread = None

    def set_command_callback(self, callback):
        """Set callback function for commands: callback(command_text)"""
        self.callback = callback

    def _audio_callback(self, indata, frames, time_info, status):
        """Process incoming audio."""
        if status:
            pass  # Ignore status messages

        audio = indata.copy().flatten()
        rms = np.sqrt(np.mean(audio.astype(float) ** 2))
        now = time.time()

        if rms > SPEECH_THRESHOLD:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start = now
                self._audio_buffer = []
            self._last_speech = now
            self._audio_buffer.append(audio.copy())

        elif self._is_speaking:
            self._audio_buffer.append(audio.copy())

            # Check if speech ended
            if self._last_speech and (now - self._last_speech) > SILENCE_AFTER_SPEECH:
                self._process_speech()
            elif self._speech_start and (now - self._speech_start) > MAX_SPEECH_DURATION:
                self._process_speech()

    def _process_speech(self):
        """Process captured speech with Whisper API."""
        if not self._audio_buffer:
            self._reset_state()
            return

        duration = len(self._audio_buffer) * 0.5  # Approximate
        if duration < MIN_SPEECH_DURATION:
            self._reset_state()
            return

        # Combine audio
        audio_data = np.concatenate(self._audio_buffer)

        # Convert to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.astype(np.int16).tobytes())

        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"

        # Transcribe with Whisper
        try:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer
            )
            text = transcript.text.strip().lower()

            if text:
                self._handle_transcription(text)

        except Exception as e:
            print(f"[Voice] Whisper error: {e}")

        self._reset_state()

    def _reset_state(self):
        """Reset audio capture state."""
        self._audio_buffer = []
        self._is_speaking = False
        self._speech_start = None
        self._last_speech = None

    def _handle_transcription(self, text):
        """Handle transcribed text."""
        # Check for wake word
        has_wake = False
        command = text

        for wake in WAKE_WORDS:
            if wake in text:
                has_wake = True
                # Extract command after wake word
                parts = text.split(wake, 1)
                if len(parts) > 1:
                    command = parts[1].strip()
                else:
                    command = ""
                break

        if has_wake or self._always_listen:
            if command:
                print(f"[Voice] Command: \"{command}\"")
                self.command_queue.put(command)
                if self.callback:
                    self.callback(command)

    @property
    def _always_listen(self):
        """For now, always process speech (no wake word required)."""
        return True

    def start(self):
        """Start listening."""
        if self.running:
            return

        self.running = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=int(SAMPLE_RATE * 0.5),
            callback=self._audio_callback
        )
        self._stream.start()
        print("[Voice] Listening for commands...")

    def stop(self):
        """Stop listening."""
        self.running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_command(self, timeout=None):
        """Get next command (blocking)."""
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_command_nowait(self):
        """Get command without blocking."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None


# Command parser for robot
COMMANDS = {
    'stop': ['stop', 'halt', 'freeze', 'wait'],
    'come': ['come here', 'come to me', 'come back'],
    'turn_left': ['turn left', 'go left', 'left'],
    'turn_right': ['turn right', 'go right', 'right'],
    'turn_around': ['turn around', 'one eighty', '180', 'about face'],
    'forward': ['go forward', 'move forward', 'forward', 'go ahead'],
    'backward': ['go back', 'move back', 'backward', 'reverse', 'back up'],
    'explore': ['explore', 'keep exploring', 'continue', 'go explore'],
    'what_see': ['what do you see', 'what is there', 'describe', 'look around'],
    'where': ['where are you', 'location', 'position'],
    'status': ['status', 'how are you', 'report'],
    'dance': ['dance', 'spin', 'do a dance'],
    'quiet': ['be quiet', 'shut up', 'silence', 'shush'],
    'speak': ['say something', 'talk', 'speak'],
}


def parse_command(text):
    """Parse text into a command."""
    text = text.lower().strip()

    for cmd, phrases in COMMANDS.items():
        for phrase in phrases:
            if phrase in text:
                return cmd

    return None


def test_voice():
    """Test voice commander."""
    print("=" * 50)
    print("VOICE COMMANDER TEST")
    print("Speak commands like 'stop', 'turn left', etc.")
    print("Press Ctrl+C to quit")
    print("=" * 50)

    commander = VoiceCommander()

    def on_command(text):
        cmd = parse_command(text)
        if cmd:
            print(f"  -> Parsed: {cmd}")
        else:
            print(f"  -> Unknown command")

    commander.set_command_callback(on_command)
    commander.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        commander.stop()


if __name__ == "__main__":
    test_voice()
