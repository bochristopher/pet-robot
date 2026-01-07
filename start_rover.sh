#!/bin/bash
source /opt/ros/humble/setup.bash
source ~/rover_ws/install/setup.bash

# Kill any existing processes
pkill -f rplidar 2>/dev/null
pkill -f lidar_filter 2>/dev/null
pkill -f motor_controller 2>/dev/null
pkill -f camera_node 2>/dev/null
pkill -f foxglove 2>/dev/null
sleep 2

echo "Starting motor controller..."
ros2 run rover_bringup motor_controller --ros-args -p serial_port:=/dev/ttyACM0 &
sleep 3

echo "Starting LIDAR..."
ros2 run rplidar_ros rplidar_node --ros-args -p serial_port:=/dev/ttyUSB0 -p serial_baudrate:=115200 -p frame_id:=laser &
sleep 3

echo "Starting LIDAR filter..."
ros2 run rover_bringup lidar_filter &
sleep 1

echo "Starting OAK-1 camera..."
/home/bo/.local/bin/oak_camera_node &
sleep 2

echo "Starting IMU..."
ros2 run rover_bringup imu_node --ros-args -p i2c_bus:=7 &
sleep 1

echo "Starting Foxglove bridge..."
ros2 run foxglove_bridge foxglove_bridge &
sleep 2

echo "Starting TF publishers..."
ros2 run tf2_ros static_transform_publisher 0 0 0.1 3.14159 0 0 base_link laser &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_link &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &

echo ""
echo "All started! Connect to ws://192.168.1.70:8765"
echo "Press Ctrl+C to stop all"
wait
