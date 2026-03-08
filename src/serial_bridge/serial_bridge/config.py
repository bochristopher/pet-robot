"""Central configuration: topics, message types, defaults, ROS2 parameters."""

from __future__ import annotations

from dataclasses import dataclass

# ── ROS2 topic names ──────────────────────────────────────────
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_MOTOR_ENABLE = "/motor_enable"
TOPIC_WHEEL_ENCODERS = "/wheel_encoders"
TOPIC_BATTERY_VOLTAGE = "/battery_voltage"
TOPIC_IMU_RAW = "/imu/data_raw"
TOPIC_ODOM = "/odom"
TOPIC_BUMPER = "/bumper"
TOPIC_BRIDGE_STATUS = "/bridge_status"
TOPIC_DIAGNOSTICS = "/diagnostics"

# ── Serial protocol message type chars ────────────────────────
MSG_MOTOR = "M"
MSG_SERVO = "S"
MSG_LED = "L"
MSG_ENABLE = "E"
MSG_DISABLE = "D"
MSG_RESET_ENC = "R"
MSG_FW_VERSION = "V"
MSG_HEARTBEAT = "H"

MSG_ENCODERS = "E"
MSG_BATTERY = "B"
MSG_IMU = "I"
MSG_BUMPER = "K"
MSG_HB_ACK = "A"
MSG_FW_RESP = "F"


@dataclass
class BridgeConfig:
    """All tuneable parameters with sane defaults."""

    serial_port: str = "/dev/ttyACM0"
    baud_rate: int = 115_200
    reconnect_interval: float = 2.0

    cmd_vel_timeout: float = 0.5
    heartbeat_interval: float = 0.5
    heartbeat_timeout: float = 1.5
    max_checksum_errors: int = 10

    # Mecanum geometry (metres)
    wheel_base: float = 0.20
    track_width: float = 0.22
    wheel_radius: float = 0.03
    encoder_cpr: int = 1440

    max_motor_speed: int = 255

    publish_rate_fast: float = 50.0  # Hz — encoders / IMU
    publish_rate_slow: float = 2.0   # Hz — battery / bumper / diagnostics

    odom_frame_id: str = "odom"
    base_frame_id: str = "base_link"

    @property
    def rotation_arm(self) -> float:
        return (self.wheel_base + self.track_width) / 2.0

    @property
    def fast_period(self) -> float:
        return 1.0 / self.publish_rate_fast

    @property
    def slow_period(self) -> float:
        return 1.0 / self.publish_rate_slow

    @property
    def metres_per_tick(self) -> float:
        from math import pi
        return (2.0 * pi * self.wheel_radius) / self.encoder_cpr


# (parameter_name, config_attr, python_type)
PARAM_DECLARATIONS: list[tuple[str, str, type]] = [
    ("serial_port", "serial_port", str),
    ("baud_rate", "baud_rate", int),
    ("reconnect_interval", "reconnect_interval", float),
    ("cmd_vel_timeout", "cmd_vel_timeout", float),
    ("heartbeat_interval", "heartbeat_interval", float),
    ("heartbeat_timeout", "heartbeat_timeout", float),
    ("max_checksum_errors", "max_checksum_errors", int),
    ("wheel_base", "wheel_base", float),
    ("track_width", "track_width", float),
    ("wheel_radius", "wheel_radius", float),
    ("encoder_cpr", "encoder_cpr", int),
    ("max_motor_speed", "max_motor_speed", int),
    ("publish_rate_fast", "publish_rate_fast", float),
    ("publish_rate_slow", "publish_rate_slow", float),
    ("odom_frame_id", "odom_frame_id", str),
    ("base_frame_id", "base_frame_id", str),
]
