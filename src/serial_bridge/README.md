# serial_bridge

ROS2 Humble Python package that bridges `/cmd_vel` and sensor topics to an Arduino Mega 2560 over USB serial. Designed for a mecanum-wheeled pet rover running on a Jetson Orin Nano.

## Architecture

```
teleop / Nav2                serial_bridge              Arduino Mega
───────────────            ─────────────────          ────────────────
 /cmd_vel (Twist)  ──►  bridge_node.py  ──USB──►  mega_bridge_firmware
                        protocol.py                    L298N motors
 /wheel_encoders   ◄──  serial_handler.py ◄──USB──  encoders / IMU
 /odom             ◄──  mecanum_kinematics.py         battery / bumper
 /battery_voltage  ◄──  safety.py
 /imu/data_raw     ◄──
 /bridge_status    ◄──
```

## Quick start

```bash
# Build
cd ~/rover_ws
colcon build --packages-select serial_bridge
source install/setup.bash

# Launch with defaults
ros2 launch serial_bridge bridge.launch.py

# Or with a different port
ros2 launch serial_bridge bridge.launch.py serial_port:=/dev/ttyUSB0
```

## Topics

| Topic | Type | Direction |
|-------|------|-----------|
| `/cmd_vel` | `geometry_msgs/Twist` | Subscribe |
| `/motor_enable` | `std_msgs/Bool` | Subscribe |
| `/wheel_encoders` | `std_msgs/Int32MultiArray` | Publish |
| `/battery_voltage` | `std_msgs/Float32` | Publish |
| `/imu/data_raw` | `sensor_msgs/Imu` | Publish |
| `/odom` | `nav_msgs/Odometry` | Publish |
| `/bumper` | `std_msgs/Bool` | Publish |
| `/bridge_status` | `std_msgs/String` | Publish |
| `/diagnostics` | `diagnostic_msgs/DiagnosticStatus` | Publish |

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/reset_encoders` | `std_srvs/Trigger` | Zero encoder counts and odometry |
| `/estop` | `std_srvs/Trigger` | Emergency stop — disable motors |
| `/clear_estop` | `std_srvs/Trigger` | Re-enable motor commands |

## Parameters

See `config/bridge_params.yaml` for all tuneable parameters.

## Serial protocol

Wire format (both directions): `<TYPE>:<fields>,<XOR_HEX>\n`

## Running tests

```bash
cd ~/rover_ws
colcon test --packages-select serial_bridge
colcon test-result --verbose
```
