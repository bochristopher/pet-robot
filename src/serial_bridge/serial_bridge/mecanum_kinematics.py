"""
Mecanum wheel kinematics + incremental odometry.

Standard mecanum layout (top-down, rollers at 45°):
    FL ╲    ╱ FR
    RL ╱    ╲ RR

Inverse kinematics (body → wheel angular velocities):
    ω_FL = (vx - vy - (L+W)/2 · ωz) / R
    ω_FR = (vx + vy + (L+W)/2 · ωz) / R
    ω_RL = (vx + vy - (L+W)/2 · ωz) / R
    ω_RR = (vx - vy + (L+W)/2 · ωz) / R

Forward kinematics (wheel → body velocities):
    vx = R/4 · (ω_FL + ω_FR + ω_RL + ω_RR)
    vy = R/4 · (-ω_FL + ω_FR + ω_RL - ω_RR)
    ωz = R / (4·k) · (-ω_FL + ω_FR - ω_RL + ω_RR)
where k = (L + W) / 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def inverse_kinematics(
    vx: float, vy: float, wz: float,
    wheel_radius: float, rotation_arm: float,
) -> tuple[float, float, float, float]:
    """Body twist → per-wheel angular velocities (rad/s)."""
    inv_r = 1.0 / wheel_radius
    fl = inv_r * (vx - vy - rotation_arm * wz)
    fr = inv_r * (vx + vy + rotation_arm * wz)
    rl = inv_r * (vx + vy - rotation_arm * wz)
    rr = inv_r * (vx - vy + rotation_arm * wz)
    return fl, fr, rl, rr


def forward_kinematics(
    w_fl: float, w_fr: float, w_rl: float, w_rr: float,
    wheel_radius: float, rotation_arm: float,
) -> tuple[float, float, float]:
    """Per-wheel angular velocities → body (vx, vy, wz)."""
    r4 = wheel_radius / 4.0
    vx = r4 * (w_fl + w_fr + w_rl + w_rr)
    vy = r4 * (-w_fl + w_fr + w_rl - w_rr)
    wz = r4 / rotation_arm * (-w_fl + w_fr - w_rl + w_rr)
    return vx, vy, wz


def wheel_speed_to_pwm(
    omega: float,
    max_omega: float,
    max_pwm: int = 255,
) -> int:
    """Map angular velocity (rad/s) → integer PWM in [-max_pwm, max_pwm]."""
    if max_omega <= 0.0:
        return 0
    ratio = omega / max_omega
    ratio = max(-1.0, min(1.0, ratio))
    return int(round(ratio * max_pwm))


def twist_to_pwm(
    vx: float, vy: float, wz: float,
    wheel_radius: float, rotation_arm: float,
    max_pwm: int = 255,
    max_wheel_speed: float | None = None,
) -> tuple[int, int, int, int]:
    """Full pipeline: body twist → 4 integer PWM values, clamped & normalised."""
    fl, fr, rl, rr = inverse_kinematics(vx, vy, wz, wheel_radius, rotation_arm)

    speeds = [fl, fr, rl, rr]
    max_abs = max(abs(s) for s in speeds) if speeds else 0.0

    if max_wheel_speed is not None and max_wheel_speed > 0:
        ref = max_wheel_speed
    elif max_abs > 0:
        ref = max_abs
    else:
        return 0, 0, 0, 0

    # Normalise so the fastest wheel is at max_pwm
    if max_abs > ref:
        scale = ref / max_abs
        speeds = [s * scale for s in speeds]
        ref_norm = ref
    else:
        ref_norm = ref

    pwms = [wheel_speed_to_pwm(s, ref_norm, max_pwm) for s in speeds]
    return pwms[0], pwms[1], pwms[2], pwms[3]


# ── Incremental Odometry ──────────────────────────────────────

@dataclass
class OdometryState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    prev_ticks: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    initialised: bool = False

    def update(
        self,
        ticks: list[int],
        dt: float,
        metres_per_tick: float,
        wheel_radius: float,
        rotation_arm: float,
    ) -> None:
        if not self.initialised:
            self.prev_ticks = list(ticks)
            self.initialised = True
            return

        if dt <= 0:
            return

        deltas = [t - p for t, p in zip(ticks, self.prev_ticks)]
        self.prev_ticks = list(ticks)

        # Tick deltas → angular velocities
        omega = [
            (d * metres_per_tick) / (wheel_radius * dt) if dt > 0 else 0.0
            for d in deltas
        ]

        vx, vy, wz = forward_kinematics(
            omega[0], omega[1], omega[2], omega[3],
            wheel_radius, rotation_arm,
        )

        self.vx = vx
        self.vy = vy
        self.wz = wz

        # Integrate in the body frame, then rotate to world
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        dx = (vx * cos_t - vy * sin_t) * dt
        dy = (vx * sin_t + vy * cos_t) * dt
        dtheta = wz * dt

        self.x += dx
        self.y += dy
        self.theta += dtheta
        # Keep theta in [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
