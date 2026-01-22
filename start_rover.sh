#!/bin/bash

# Pet Robot Startup Script
# ROS2 Humble on Jetson with Mecanum Drive

# Force consistent ROS domain
export ROS_DOMAIN_ID=0

source /opt/ros/humble/setup.bash
source ~/rover_ws/install/setup.bash

# Get local IP for display
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "=========================================="
echo "  Pet Robot - ROS2 Bringup"
echo "=========================================="

# Kill any existing processes
echo "Stopping existing processes..."
pkill -f rplidar 2>/dev/null
pkill -f lidar_filter 2>/dev/null
pkill -f motor_controller 2>/dev/null
pkill -f odometry_node 2>/dev/null
pkill -f oak_camera 2>/dev/null
pkill -f imu_node 2>/dev/null
pkill -f foxglove 2>/dev/null
sleep 2

echo "[1/7] Starting motor controller..."
ros2 run rover_bringup motor_controller --ros-args -p serial_port:=/dev/ttyACM0 &
sleep 3

echo "[2/7] Starting LIDAR..."
ros2 run rplidar_ros rplidar_node --ros-args \
    -p serial_port:=/dev/ttyUSB0 \
    -p serial_baudrate:=115200 \
    -p frame_id:=laser &
sleep 3

echo "[3/7] Starting LIDAR filter..."
ros2 run rover_bringup lidar_filter &
sleep 1

echo "[4/7] Starting OAK-D camera..."
/home/bo/.local/bin/oak_camera_node &
sleep 2

echo "[5/7] Starting IMU (I2C bus 7)..."
ros2 run rover_bringup imu_node --ros-args -p i2c_bus:=7 &
sleep 1

echo "[6/7] Starting odometry..."
ros2 run rover_bringup odometry_node --ros-args \
    -p wheel_radius:=0.0381 \
    -p wheel_base:=0.2286 \
    -p publish_tf:=true &
sleep 1

echo "[7/7] Starting Foxglove bridge..."
ros2 run foxglove_bridge foxglove_bridge &
sleep 2

echo ""
echo "Starting TF publishers..."
# LIDAR is rotated 180° (mounted backwards)
ros2 run tf2_ros static_transform_publisher 0 0 0.1 3.14159 0 0 base_link laser &
# map -> odom (will be updated by AMCL/SLAM when available)
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &

echo ""
echo "=========================================="
echo "  All nodes started!"
echo "  Foxglove: ws://${LOCAL_IP}:8765"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop all"
wait
