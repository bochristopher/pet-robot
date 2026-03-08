"""Unit tests for serial_bridge.protocol."""

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
    encode_request_version,
    make_frame,
    parse_frame,
    xor_checksum,
)


class TestChecksum:
    def test_empty(self) -> None:
        assert xor_checksum("") == 0

    def test_single_char(self) -> None:
        assert xor_checksum("A") == ord("A")

    def test_round_trip(self) -> None:
        data = "M:100,-100,50,-50"
        cs = xor_checksum(data)
        assert 0 <= cs <= 255


class TestFraming:
    def test_make_and_parse(self) -> None:
        payload = "M:100,-100,50,-50"
        frame = make_frame(payload)
        assert frame.endswith("\n")
        parsed = parse_frame(frame)
        assert parsed == payload

    def test_bad_checksum(self) -> None:
        frame = "M:1,2,3,4,FF\n"
        assert parse_frame(frame) is None

    def test_too_short(self) -> None:
        assert parse_frame("AB") is None

    def test_no_comma(self) -> None:
        assert parse_frame("ABCDE") is None


class TestEncoders:
    def test_encode_motor(self) -> None:
        frame = encode_motor(100, -100, 50, -50)
        body = parse_frame(frame)
        assert body is not None
        assert body.startswith("M:")

    def test_encode_enable_disable(self) -> None:
        e = parse_frame(encode_enable())
        d = parse_frame(encode_disable())
        assert e is not None and e.startswith("E:")
        assert d is not None and d.startswith("D:")

    def test_encode_heartbeat(self) -> None:
        h = parse_frame(encode_heartbeat())
        assert h is not None and h.startswith("H:")

    def test_encode_reset(self) -> None:
        r = parse_frame(encode_reset_encoders())
        assert r is not None and r.startswith("R:")

    def test_encode_version(self) -> None:
        v = parse_frame(encode_request_version())
        assert v is not None and v.startswith("V:")


class TestDecoders:
    def test_decode_encoders(self) -> None:
        result = decode_encoders("E:10,20,30,40")
        assert result == (10, 20, 30, 40)

    def test_decode_encoders_negative(self) -> None:
        result = decode_encoders("E:-5,10,-15,20")
        assert result == (-5, 10, -15, 20)

    def test_decode_encoders_bad(self) -> None:
        assert decode_encoders("E:1,2,3") is None
        assert decode_encoders("X:1,2,3,4") is None

    def test_decode_battery(self) -> None:
        assert decode_battery("B:12.45") == 12.45

    def test_decode_battery_bad(self) -> None:
        assert decode_battery("B:abc") is None
        assert decode_battery("X:12.0") is None

    def test_decode_imu(self) -> None:
        result = decode_imu("I:0.0,0.0,9.81,0.1,0.2,0.3")
        assert result is not None
        assert len(result) == 6
        assert abs(result[2] - 9.81) < 0.001

    def test_decode_bumper(self) -> None:
        assert decode_bumper("K:1") == 1
        assert decode_bumper("K:0") == 0

    def test_decode_heartbeat_ack(self) -> None:
        assert decode_heartbeat_ack("A:42") == 42

    def test_decode_firmware(self) -> None:
        assert decode_firmware("F:mega-usb-v3.0") == "mega-usb-v3.0"
