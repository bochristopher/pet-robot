"""
Serial protocol: framing, checksum, encode & decode.

Wire format (both directions):
    <TYPE>:<fields>,<XOR_HEX>\n

XOR checksum covers everything from TYPE through the last field character
(excluding the final comma + hex + newline).
"""

from __future__ import annotations


def xor_checksum(data: str) -> int:
    c = 0
    for ch in data:
        c ^= ord(ch)
    return c


def make_frame(payload: str) -> str:
    cs = xor_checksum(payload)
    return f"{payload},{cs:02X}\n"


def parse_frame(line: str) -> str | None:
    """Validate checksum and return raw payload, or None on failure."""
    line = line.strip()
    if len(line) < 4:
        return None
    cs_hex = line[-2:]
    body = line[:-3]  # everything before ",XX"
    if line[-3] != ",":
        return None
    try:
        expected = int(cs_hex, 16)
    except ValueError:
        return None
    if xor_checksum(body) != expected:
        return None
    return body


# ── Encoder helpers ───────────────────────────────────────────

def encode_motor(fl: int, fr: int, rl: int, rr: int) -> str:
    return make_frame(f"M:{fl},{fr},{rl},{rr}")


def encode_enable() -> str:
    return make_frame("E:")


def encode_disable() -> str:
    return make_frame("D:")


def encode_heartbeat() -> str:
    return make_frame("H:")


def encode_reset_encoders() -> str:
    return make_frame("R:")


def encode_request_version() -> str:
    return make_frame("V:")


def encode_servo(channel: int, angle: int) -> str:
    return make_frame(f"S:{channel},{angle}")


def encode_led(r: int, g: int, b: int) -> str:
    return make_frame(f"L:{r},{g},{b}")


# ── Decode helpers ────────────────────────────────────────────

def decode_encoders(body: str) -> tuple[int, int, int, int] | None:
    """Parse 'E:fl,fr,rl,rr' → (fl, fr, rl, rr)."""
    if not body.startswith("E:"):
        return None
    parts = body[2:].split(",")
    if len(parts) != 4:
        return None
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def decode_battery(body: str) -> float | None:
    if not body.startswith("B:"):
        return None
    try:
        return float(body[2:])
    except ValueError:
        return None


def decode_imu(body: str) -> tuple[float, ...] | None:
    """Parse 'I:ax,ay,az,gx,gy,gz' → 6-tuple."""
    if not body.startswith("I:"):
        return None
    parts = body[2:].split(",")
    if len(parts) != 6:
        return None
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        return None


def decode_bumper(body: str) -> int | None:
    if not body.startswith("K:"):
        return None
    try:
        return int(body[2:])
    except ValueError:
        return None


def decode_heartbeat_ack(body: str) -> int | None:
    if not body.startswith("A:"):
        return None
    try:
        return int(body[2:])
    except ValueError:
        return None


def decode_firmware(body: str) -> str | None:
    if not body.startswith("F:"):
        return None
    return body[2:]
