"""
Serial handler with polling-based reads (no background thread).

Provides:
  - poll_read() to be called from a timer — reads all available data.
  - Thread-safe send() for outbound frames.
  - Automatic reconnection on disconnect / error.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import serial  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class SerialHandler:
    def __init__(
        self,
        port: str,
        baud: int,
        on_connect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        self._port = port
        self._baud = baud
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect

        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self._rx_buf = ""

    @property
    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def try_connect(self) -> bool:
        """Attempt to open the serial port. Returns True on success."""
        if self.connected:
            return True
        try:
            self._ser = serial.Serial(
                self._port, self._baud,
                timeout=0,  # non-blocking reads
                write_timeout=0.5,
            )
            import time
            time.sleep(2)  # wait for Mega to boot after DTR reset
            self._ser.reset_input_buffer()
            self._rx_buf = ""
            logger.info("Connected to %s @ %d", self._port, self._baud)
            if self._on_connect:
                self._on_connect()
            return True
        except (serial.SerialException, OSError) as exc:
            logger.debug("Cannot open %s: %s", self._port, exc)
            self._ser = None
            return False

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                if self._on_disconnect:
                    self._on_disconnect()

    def send(self, frame: str) -> bool:
        """Thread-safe write to serial port."""
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                return False
            try:
                self._ser.write(frame.encode("ascii"))
                return True
            except (serial.SerialException, OSError):
                return False

    def poll_read(self) -> list[str]:
        """Read all available complete lines from the serial port.

        Call this from a fast timer callback. Returns a list of
        complete lines (stripped, no newline).
        """
        if not self.connected:
            return []

        try:
            available = self._ser.in_waiting  # type: ignore[union-attr]
            if available <= 0:
                return []
            raw = self._ser.read(available)  # type: ignore[union-attr]
            if not raw:
                return []
        except (serial.SerialException, OSError):
            self.close()
            return []

        self._rx_buf += raw.decode("ascii", errors="replace")

        lines: list[str] = []
        while "\n" in self._rx_buf:
            line, self._rx_buf = self._rx_buf.split("\n", 1)
            line = line.strip()
            if line:
                lines.append(line)

        # Prevent buffer from growing unbounded on partial data
        if len(self._rx_buf) > 512:
            self._rx_buf = ""

        return lines
