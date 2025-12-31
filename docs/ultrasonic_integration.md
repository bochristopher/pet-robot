# HC-SR04 Ultrasonic Sensor Integration

## Overview
Three HC-SR04 ultrasonic sensors integrated into Arduino Mega-controlled robot car for near-field obstacle detection and collision avoidance.

## Hardware Configuration

### Sensor Placement
| Sensor | Position | Mount Height | Orientation |
|--------|----------|--------------|-------------|
| Front Left | Front chassis, left side | 12-15 cm | Level, angled slightly outward |
| Front Right | Front chassis, right side | 12-15 cm | Level, angled slightly outward |
| Rear | Rear chassis, center | 12-15 cm | Level, facing backward |

### Power
- **VCC**: 5V from Arduino Mega
- **GND**: Common ground with Arduino

### Arduino Pin Mapping

| Sensor | Signal | Arduino Pin |
|--------|--------|-------------|
| Front Left | TRIG | 48 |
| Front Left | ECHO | 49 |
| Front Right | TRIG | 52 |
| Front Right | ECHO | 53 |
| Rear | TRIG | 50 |
| Rear | ECHO | 51 |

## Software Interface

### Arduino Firmware
The Arduino handles ultrasonic sensor readings via the `ULTRASONIC` command.
Response format: `US:FL,FR,BACK` (values in cm, -1 for timeout)

### Python Driver

**Driver**: `/home/bo/robot_pet/ultrasonic.py`

```python
from ultrasonic import UltrasonicSensors

# Standalone connection
sensors = UltrasonicSensors()

# Or share existing Arduino serial connection
sensors = UltrasonicSensors(serial_conn=arduino)

reading = sensors.read_all()

print(f"Front Left:  {reading.front_left:.1f} cm")
print(f"Front Right: {reading.front_right:.1f} cm")
print(f"Rear:        {reading.back:.1f} cm")

sensors.close()
```

### Key Functions
- `read_all()` → Returns all three distances (cm)
- `read_front_left()` → Single sensor reading
- `read_front_right()` → Single sensor reading
- `read_back()` → Single sensor reading

## Usage

Ultrasonic sensors provide **collision avoidance**, not mapping:
- Detection range: ~2 cm to 400 cm
- Effective for close obstacles missed by LiDAR
- Triggers emergency stop/turn when obstacle < 25 cm

## Implementation Notes
- Arduino reads sensors sequentially with 5ms delay between each to prevent cross-talk
- Timeout: 30ms (~5m max range)
- Python driver shares serial connection with motor control to avoid port conflicts

## Dependencies
- `pyserial` Python library (`pip install pyserial`)
- Arduino Mega with robot_motor_control firmware
