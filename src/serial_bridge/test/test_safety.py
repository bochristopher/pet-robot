"""Unit tests for serial_bridge.safety."""

import time
from unittest.mock import patch

from serial_bridge.safety import BridgeState, SafetyMonitor


class TestSafetyMonitor:
    def test_initial_state(self) -> None:
        sm = SafetyMonitor()
        assert sm.state == BridgeState.DISCONNECTED
        assert sm.is_cmd_vel_stale

    def test_connect_disconnect(self) -> None:
        sm = SafetyMonitor()
        sm.on_connect()
        assert sm.state == BridgeState.CONNECTED
        sm.on_disconnect()
        assert sm.state == BridgeState.DISCONNECTED

    def test_cmd_vel_freshness(self) -> None:
        sm = SafetyMonitor(cmd_vel_timeout=0.1)
        sm.on_connect()
        sm.on_cmd_vel()
        assert not sm.is_cmd_vel_stale
        time.sleep(0.15)
        assert sm.is_cmd_vel_stale

    def test_should_send_motors(self) -> None:
        sm = SafetyMonitor(cmd_vel_timeout=1.0, heartbeat_timeout=5.0)
        sm.on_connect()
        assert not sm.should_send_motors  # no cmd_vel yet
        sm.on_cmd_vel()
        assert sm.should_send_motors

    def test_estop(self) -> None:
        sm = SafetyMonitor(cmd_vel_timeout=10.0, heartbeat_timeout=10.0)
        sm.on_connect()
        sm.on_cmd_vel()
        assert sm.should_send_motors
        sm.trigger_estop()
        assert sm.state == BridgeState.ESTOPPED
        assert not sm.should_send_motors
        sm.clear_estop()
        assert sm.state == BridgeState.CONNECTED

    def test_mega_timeout(self) -> None:
        sm = SafetyMonitor(heartbeat_timeout=0.1)
        sm.on_connect()
        time.sleep(0.15)
        assert sm.state == BridgeState.MEGA_TIMEOUT

    def test_heartbeat_ack_resets_timeout(self) -> None:
        sm = SafetyMonitor(heartbeat_timeout=0.2)
        sm.on_connect()
        time.sleep(0.1)
        sm.on_heartbeat_ack()
        assert sm.state == BridgeState.CONNECTED

    def test_checksum_error_count(self) -> None:
        sm = SafetyMonitor()
        assert sm.checksum_errors == 0
        sm.on_checksum_error()
        sm.on_checksum_error()
        assert sm.checksum_errors == 2

    def test_connect_resets_errors(self) -> None:
        sm = SafetyMonitor()
        sm.on_checksum_error()
        sm.on_connect()
        assert sm.checksum_errors == 0
