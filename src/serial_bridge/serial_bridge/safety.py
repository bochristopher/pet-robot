"""
Safety layer: cmd_vel timeout, heartbeat watchdog, e-stop logic.

All timing is monotonic-clock based. This module is pure-logic with no ROS
or I/O dependencies so it can be unit-tested independently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class BridgeState(Enum):
    DISCONNECTED = auto()
    CONNECTED = auto()
    MEGA_TIMEOUT = auto()
    ESTOPPED = auto()


@dataclass
class SafetyMonitor:
    """Tracks cmd_vel freshness, heartbeat health, and e-stop."""

    cmd_vel_timeout: float = 0.5
    heartbeat_timeout: float = 1.5

    _last_cmd_vel_time: float = field(default=0.0, init=False, repr=False)
    _last_hb_ack_time: float = field(default=0.0, init=False, repr=False)
    _connect_time: float = field(default=0.0, init=False, repr=False)
    _estopped: bool = field(default=False, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)
    _checksum_errors: int = field(default=0, init=False, repr=False)

    # ── External events ───────────────────────────────────────

    def on_connect(self) -> None:
        now = time.monotonic()
        self._connected = True
        self._connect_time = now
        self._last_hb_ack_time = now
        self._last_cmd_vel_time = 0.0
        self._estopped = False
        self._checksum_errors = 0

    def on_disconnect(self) -> None:
        self._connected = False

    def on_cmd_vel(self) -> None:
        self._last_cmd_vel_time = time.monotonic()

    def on_heartbeat_ack(self) -> None:
        self._last_hb_ack_time = time.monotonic()

    def on_checksum_error(self) -> None:
        self._checksum_errors += 1

    def trigger_estop(self) -> None:
        self._estopped = True

    def clear_estop(self) -> None:
        self._estopped = False
        self._last_cmd_vel_time = 0.0

    # ── Queries ───────────────────────────────────────────────

    @property
    def state(self) -> BridgeState:
        if self._estopped:
            return BridgeState.ESTOPPED
        if not self._connected:
            return BridgeState.DISCONNECTED
        if self._is_mega_timeout():
            return BridgeState.MEGA_TIMEOUT
        return BridgeState.CONNECTED

    @property
    def is_cmd_vel_stale(self) -> bool:
        if self._last_cmd_vel_time == 0.0:
            return True
        return (time.monotonic() - self._last_cmd_vel_time) > self.cmd_vel_timeout

    @property
    def should_send_motors(self) -> bool:
        return (
            self._connected
            and not self._estopped
            and not self.is_cmd_vel_stale
            and not self._is_mega_timeout()
        )

    @property
    def checksum_errors(self) -> int:
        return self._checksum_errors

    def _is_mega_timeout(self) -> bool:
        if not self._connected:
            return False
        return (time.monotonic() - self._last_hb_ack_time) > self.heartbeat_timeout
