"""
Main ROS2 node: bridges /cmd_vel <-> serial protocol <-> Arduino Mega.

Uses polling-based serial reads from a fast ROS2 timer, keeping all I/O
in the single-threaded executor. No background threads needed.
"""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, Int32MultiArray, String
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from std_srvs.srv import Trigger

from serial_bridge.config import (
    BridgeConfig,
    PARAM_DECLARATIONS,
    TOPIC_BATTERY_VOLTAGE,
    TOPIC_BRIDGE_STATUS,
    TOPIC_BUMPER,
    TOPIC_CMD_VEL,
    TOPIC_DIAGNOSTICS,
    TOPIC_IMU_RAW,
    TOPIC_MOTOR_ENABLE,
    TOPIC_ODOM,
    TOPIC_WHEEL_ENCODERS,
)
from serial_bridge.mecanum_kinematics import OdometryState, twist_to_pwm
from serial_bridge.protocol import (
    decode_battery,
    decode_bumper,
    decode_encoders,
    decode_firmware,
    decode_heartbeat_ack,
    decode_imu,
    encode_disable,
    encode_enable,
    encode_heartbeat,
    encode_motor,
    encode_reset_encoders,
    parse_frame,
)
from serial_bridge.safety import BridgeState, SafetyMonitor
from serial_bridge.serial_handler import SerialHandler


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class SerialBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("serial_bridge")
        self.cfg = BridgeConfig()
        self._declare_and_load_params()

        self.safety = SafetyMonitor(
            cmd_vel_timeout=self.cfg.cmd_vel_timeout,
            heartbeat_timeout=self.cfg.heartbeat_timeout,
        )
        self.odom = OdometryState()
        self._last_encoder_time: float = 0.0
        self._fw_version: str = "unknown"

        # ── Serial handler (polling, no threads) ──────────────
        self.serial = SerialHandler(
            port=self.cfg.serial_port,
            baud=self.cfg.baud_rate,
            on_connect=self._on_serial_connect,
            on_disconnect=self._on_serial_disconnect,
        )

        # ── Publishers ────────────────────────────────────────
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_encoders = self.create_publisher(Int32MultiArray, TOPIC_WHEEL_ENCODERS, 10)
        self.pub_battery = self.create_publisher(Float32, TOPIC_BATTERY_VOLTAGE, qos_reliable)
        self.pub_imu = self.create_publisher(Imu, TOPIC_IMU_RAW, 10)
        self.pub_odom = self.create_publisher(Odometry, TOPIC_ODOM, 10)
        self.pub_bumper = self.create_publisher(Bool, TOPIC_BUMPER, qos_reliable)
        self.pub_status = self.create_publisher(String, TOPIC_BRIDGE_STATUS, qos_reliable)
        self.pub_diag = self.create_publisher(DiagnosticStatus, TOPIC_DIAGNOSTICS, qos_reliable)

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(Twist, TOPIC_CMD_VEL, self._cb_cmd_vel, 10)
        self.create_subscription(Bool, TOPIC_MOTOR_ENABLE, self._cb_motor_enable, 10)

        # ── Services ──────────────────────────────────────────
        self.create_service(Trigger, "reset_encoders", self._srv_reset_encoders)
        self.create_service(Trigger, "estop", self._srv_estop)
        self.create_service(Trigger, "clear_estop", self._srv_clear_estop)

        # ── Timers ────────────────────────────────────────────
        self.create_timer(0.01, self._timer_poll_serial)  # 100 Hz serial poll
        self.create_timer(self.cfg.heartbeat_interval, self._timer_heartbeat)
        self.create_timer(self.cfg.slow_period, self._timer_slow)
        self.create_timer(0.05, self._timer_cmd_vel_watchdog)  # 20 Hz

        # ── Connect immediately ───────────────────────────────
        self.serial.try_connect()
        self.get_logger().info(
            f"SerialBridge starting — port={self.cfg.serial_port} baud={self.cfg.baud_rate}"
        )

    # ── Parameter loading ─────────────────────────────────────

    def _declare_and_load_params(self) -> None:
        for ros_name, attr, _ in PARAM_DECLARATIONS:
            default = getattr(self.cfg, attr)
            self.declare_parameter(ros_name, default)
            val = self.get_parameter(ros_name).value
            setattr(self.cfg, attr, val)

    # ── Serial events ─────────────────────────────────────────

    def _on_serial_connect(self) -> None:
        self.safety.on_connect()
        self.serial.send(encode_enable())
        self.get_logger().info("Serial connected — motors enabled")

    def _on_serial_disconnect(self) -> None:
        self.safety.on_disconnect()
        self.get_logger().warn("Serial disconnected — will reconnect")

    # ── Fast serial poll timer ────────────────────────────────

    def _timer_poll_serial(self) -> None:
        if not self.serial.connected:
            self.serial.try_connect()
            return

        lines = self.serial.poll_read()
        for raw_line in lines:
            body = parse_frame(raw_line)
            if body is None:
                self.safety.on_checksum_error()
                continue
            self._dispatch_mega_message(body)

    # ── Message dispatch ──────────────────────────────────────

    def _dispatch_mega_message(self, body: str) -> None:
        if len(body) < 2 or body[1] != ":":
            return
        msg_type = body[0]

        if msg_type == "E":
            enc = decode_encoders(body)
            if enc is None:
                return
            msg = Int32MultiArray()
            msg.data = list(enc)
            self.pub_encoders.publish(msg)
            self._update_odometry(enc)

        elif msg_type == "B":
            volts = decode_battery(body)
            if volts is not None:
                self.pub_battery.publish(Float32(data=volts))

        elif msg_type == "I":
            vals = decode_imu(body)
            if vals is not None:
                imu_msg = Imu()
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = self.cfg.base_frame_id
                imu_msg.linear_acceleration.x = vals[0]
                imu_msg.linear_acceleration.y = vals[1]
                imu_msg.linear_acceleration.z = vals[2]
                imu_msg.angular_velocity.x = vals[3]
                imu_msg.angular_velocity.y = vals[4]
                imu_msg.angular_velocity.z = vals[5]
                self.pub_imu.publish(imu_msg)

        elif msg_type == "K":
            state = decode_bumper(body)
            if state is not None:
                self.pub_bumper.publish(Bool(data=(state != 0)))

        elif msg_type == "A":
            seq = decode_heartbeat_ack(body)
            if seq is not None:
                self.safety.on_heartbeat_ack()

        elif msg_type == "F":
            fw = decode_firmware(body)
            if fw is not None:
                self._fw_version = fw
                self.get_logger().info(f"Mega firmware: {fw}")

    # ── Odometry ──────────────────────────────────────────────

    def _update_odometry(self, enc: tuple[int, int, int, int]) -> None:
        now = time.monotonic()
        dt = now - self._last_encoder_time if self._last_encoder_time > 0 else 0.0
        self._last_encoder_time = now

        self.odom.update(
            ticks=list(enc),
            dt=dt,
            metres_per_tick=self.cfg.metres_per_tick,
            wheel_radius=self.cfg.wheel_radius,
            rotation_arm=self.cfg.rotation_arm,
        )

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cfg.odom_frame_id
        msg.child_frame_id = self.cfg.base_frame_id
        msg.pose.pose.position.x = self.odom.x
        msg.pose.pose.position.y = self.odom.y
        msg.pose.pose.orientation = yaw_to_quaternion(self.odom.theta)
        msg.twist.twist.linear.x = self.odom.vx
        msg.twist.twist.linear.y = self.odom.vy
        msg.twist.twist.angular.z = self.odom.wz
        self.pub_odom.publish(msg)

    # ── Subscriber callbacks ──────────────────────────────────

    def _cb_cmd_vel(self, msg: Twist) -> None:
        self.safety.on_cmd_vel()
        if not self.safety.should_send_motors:
            return
        fl, fr, rl, rr = twist_to_pwm(
            msg.linear.x, msg.linear.y, msg.angular.z,
            self.cfg.wheel_radius, self.cfg.rotation_arm,
            self.cfg.max_motor_speed,
        )
        self.serial.send(encode_motor(fl, fr, rl, rr))

    def _cb_motor_enable(self, msg: Bool) -> None:
        if msg.data:
            self.serial.send(encode_enable())
            self.get_logger().info("Motors ENABLED")
        else:
            self.serial.send(encode_disable())
            self.get_logger().info("Motors DISABLED")

    # ── Service callbacks ─────────────────────────────────────

    def _srv_reset_encoders(
        self, _req: Trigger.Request, resp: Trigger.Response,
    ) -> Trigger.Response:
        self.serial.send(encode_reset_encoders())
        self.odom = OdometryState()
        resp.success = True
        resp.message = "Encoders reset"
        return resp

    def _srv_estop(
        self, _req: Trigger.Request, resp: Trigger.Response,
    ) -> Trigger.Response:
        self.safety.trigger_estop()
        self.serial.send(encode_disable())
        self._send_zero_motors()
        resp.success = True
        resp.message = "E-STOP activated"
        self.get_logger().warn("E-STOP activated via service")
        return resp

    def _srv_clear_estop(
        self, _req: Trigger.Request, resp: Trigger.Response,
    ) -> Trigger.Response:
        self.safety.clear_estop()
        resp.success = True
        resp.message = "E-STOP cleared"
        self.get_logger().info("E-STOP cleared via service")
        return resp

    # ── Timers ────────────────────────────────────────────────

    def _timer_heartbeat(self) -> None:
        self.serial.send(encode_heartbeat())

    def _timer_cmd_vel_watchdog(self) -> None:
        if self.safety.is_cmd_vel_stale and self.safety.state == BridgeState.CONNECTED:
            self._send_zero_motors()

    def _timer_slow(self) -> None:
        state = self.safety.state
        self.pub_status.publish(String(data=state.name))

        diag = DiagnosticStatus()
        diag.name = "serial_bridge"
        diag.hardware_id = self.cfg.serial_port

        if state == BridgeState.CONNECTED:
            diag.level = DiagnosticStatus.OK
            diag.message = "OK"
        elif state == BridgeState.MEGA_TIMEOUT:
            diag.level = DiagnosticStatus.WARN
            diag.message = "Mega heartbeat timeout"
        elif state == BridgeState.ESTOPPED:
            diag.level = DiagnosticStatus.WARN
            diag.message = "E-STOP active"
        else:
            diag.level = DiagnosticStatus.ERROR
            diag.message = "Disconnected"

        diag.values = [
            KeyValue(key="firmware", value=self._fw_version),
            KeyValue(key="checksum_errors", value=str(self.safety.checksum_errors)),
            KeyValue(key="port", value=self.cfg.serial_port),
        ]
        self.pub_diag.publish(diag)

    # ── Helpers ───────────────────────────────────────────────

    def _send_zero_motors(self) -> None:
        self.serial.send(encode_motor(0, 0, 0, 0))

    def destroy_node(self) -> None:
        self.serial.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
