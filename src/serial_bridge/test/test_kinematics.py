"""Unit tests for serial_bridge.mecanum_kinematics."""

import math

from serial_bridge.mecanum_kinematics import (
    OdometryState,
    forward_kinematics,
    inverse_kinematics,
    twist_to_pwm,
)

R = 0.03       # wheel radius
K = 0.21       # rotation arm = (0.20 + 0.22) / 2


class TestInverseKinematics:
    def test_pure_forward(self) -> None:
        fl, fr, rl, rr = inverse_kinematics(1.0, 0.0, 0.0, R, K)
        assert fl > 0 and fr > 0 and rl > 0 and rr > 0
        assert abs(fl - fr) < 1e-9
        assert abs(rl - rr) < 1e-9

    def test_pure_strafe_right(self) -> None:
        fl, fr, rl, rr = inverse_kinematics(0.0, -1.0, 0.0, R, K)
        # FL and RR should be negative (strafe right), FR and RL positive
        # Actually: fl = (0 - (-1) - 0) / R = 1/R > 0
        #           fr = (0 + (-1) + 0) / R = -1/R < 0
        #           rl = (0 + (-1) - 0) / R = -1/R < 0
        #           rr = (0 - (-1) + 0) / R = 1/R > 0
        assert fl > 0
        assert fr < 0
        assert rl < 0
        assert rr > 0

    def test_pure_rotation(self) -> None:
        fl, fr, rl, rr = inverse_kinematics(0.0, 0.0, 1.0, R, K)
        assert fl < 0  # (0 - 0 - K*1) / R < 0
        assert fr > 0  # (0 + 0 + K*1) / R > 0

    def test_zero_input(self) -> None:
        fl, fr, rl, rr = inverse_kinematics(0.0, 0.0, 0.0, R, K)
        assert fl == 0 and fr == 0 and rl == 0 and rr == 0


class TestForwardKinematics:
    def test_inverse_then_forward(self) -> None:
        """Forward kinematics should recover the original twist."""
        vx_in, vy_in, wz_in = 0.5, -0.3, 0.2
        fl, fr, rl, rr = inverse_kinematics(vx_in, vy_in, wz_in, R, K)
        vx, vy, wz = forward_kinematics(fl, fr, rl, rr, R, K)
        assert abs(vx - vx_in) < 1e-9
        assert abs(vy - vy_in) < 1e-9
        assert abs(wz - wz_in) < 1e-9

    def test_all_zero(self) -> None:
        vx, vy, wz = forward_kinematics(0, 0, 0, 0, R, K)
        assert vx == 0 and vy == 0 and wz == 0


class TestTwistToPwm:
    def test_zero(self) -> None:
        assert twist_to_pwm(0, 0, 0, R, K) == (0, 0, 0, 0)

    def test_forward_symmetric(self) -> None:
        fl, fr, rl, rr = twist_to_pwm(1.0, 0.0, 0.0, R, K, 255)
        assert fl == fr == rl == rr
        assert fl > 0

    def test_clamped_to_max(self) -> None:
        fl, fr, rl, rr = twist_to_pwm(100.0, 0.0, 0.0, R, K, 255)
        assert abs(fl) <= 255
        assert abs(fr) <= 255


class TestOdometry:
    def test_initialisation(self) -> None:
        odom = OdometryState()
        assert not odom.initialised
        odom.update([0, 0, 0, 0], 0.02, 0.0001, R, K)
        assert odom.initialised
        assert odom.x == 0.0

    def test_forward_motion(self) -> None:
        odom = OdometryState()
        odom.update([0, 0, 0, 0], 0.02, 0.0001, R, K)
        odom.update([100, 100, 100, 100], 0.1, 0.0001, R, K)
        assert odom.vx > 0
        assert abs(odom.vy) < 1e-9
        assert odom.x > 0

    def test_theta_wrap(self) -> None:
        odom = OdometryState()
        odom.theta = 3.0
        odom.update([0, 0, 0, 0], 0.02, 0.0001, R, K)
        # Spin to accumulate theta beyond pi
        odom.update([0, 1000, 0, 1000], 0.1, 0.001, R, K)
        assert -math.pi <= odom.theta <= math.pi
