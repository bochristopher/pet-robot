# Pet Robot - ROS2 Rover

A ROS2 Humble rover with mecanum wheels, running on Jetson with comprehensive sensor suite.

## Hardware

- **Compute:** Jetson (Tegra)
- **Drive:** Mecanum wheels (4WD omnidirectional)
- **Controller:** Arduino Mega 2560

## Sensors

| Sensor | Description | Topics |
|--------|-------------|--------|
| **OAK-D Lite** | RGB + Stereo Depth + MobileNet-SSD object detection (Myriad X VPU) | `/camera/image_raw`, `/camera/depth/image_raw`, `/camera/detections` |
| **RPLidar** | 360° laser scanner, 10Hz | `/scan`, `/scan_filtered` |
| **MPU6050 IMU** | 6-axis accelerometer + gyroscope (I2C bus 7) | `/imu/data` |
| **Ultrasonics** | 3x HC-SR04 (front-left, front-right, back-center) | `/ultrasonic/*` |
| **Wheel Encoders** | 2x quadrature encoders | `/wheel_encoders` |

## Quick Start

```bash
# Build
cd ~/rover_ws
colcon build --packages-select rover_bringup
source install/setup.bash

# Run
~/start_rover.sh
```

## Visualization

Connect with Foxglove Studio:
```
ws://192.168.1.70:8765
```

## Nodes

| Node | Description |
|------|-------------|
| `motor_controller` | Mecanum drive control, encoder reading, ultrasonic publishing |
| `oak_camera_node` | OAK-D Lite RGB, depth, and on-device neural network inference |
| `lidar_filter` | Temporal median filter with noise removal |
| `imu_node` | MPU6050 with auto-calibration at startup |

## Calibrations Applied

- **LIDAR:** 180° rotation via TF transform
- **IMU:** Accelerometer offset and scale calibration (keep rover level at startup)
- **Encoders:** Polarity corrected (forward = positive)

## Topics

### Camera
- `/camera/image_raw` - RGB image
- `/camera/image_raw/compressed` - JPEG compressed
- `/camera/depth/image_raw` - 16-bit depth in mm
- `/camera/depth/image_colored/compressed` - Colorized depth visualization
- `/camera/detections` - Detection2DArray from MobileNet-SSD
- `/camera/detections/image/compressed` - RGB with bounding boxes

### Sensors
- `/scan` - Raw LIDAR scan
- `/scan_filtered` - Filtered LIDAR scan
- `/imu/data` - IMU with orientation, angular velocity, linear acceleration
- `/ultrasonic/front_left`, `/ultrasonic/front_right`, `/ultrasonic/back_center` - Range messages
- `/wheel_encoders` - Int32MultiArray [left, right]

### Control
- `/cmd_vel` - Twist messages for driving

## Dependencies

```bash
pip3 install depthai==2.28.0.0 smbus2
sudo apt install ros-humble-foxglove-bridge
```

## License

MIT
