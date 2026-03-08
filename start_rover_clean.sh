#!/bin/bash
# Pet Robot Clean Startup Script
# Uses ROS2 launch file for proper node lifecycle management

set -e

export ROS_DOMAIN_ID=0

source /opt/ros/humble/setup.bash
source ~/rover_ws/install/setup.bash

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "=========================================="
echo "  Pet Robot - Clean Startup"
echo "=========================================="
echo ""

# Stop ROS2 daemon and clear stale state
echo "Cleaning up previous session..."
ros2 daemon stop 2>/dev/null || true
sleep 1
rm -rf /dev/shm/fastrtps* /dev/shm/ros2* 2>/dev/null || true
ros2 daemon start
sleep 1

# Kill any orphaned processes gracefully first, then force
echo "Stopping any existing nodes..."
pkill -INT -f "micro_ros_agent|rplidar|lidar_filter|imu_node|odometry|foxglove|oak_camera|slam_toolbox" 2>/dev/null || true
sleep 2
pkill -9 -f "micro_ros_agent|rplidar|lidar_filter|imu_node|odometry|foxglove|oak_camera|slam_toolbox" 2>/dev/null || true
sleep 1

echo ""
echo "Starting OAK-D camera (separate process)..."
/home/bo/.local/bin/oak_camera_node &
OAK_PID=$!
sleep 2

echo "Starting ROS2 nodes via launch file..."
echo ""
echo "  Foxglove: ws://${LOCAL_IP}:8765"
echo ""
echo "  Press Ctrl+C to cleanly stop all nodes"
echo "=========================================="
echo ""

# Use trap to clean up OAK camera on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $OAK_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Launch with proper lifecycle management
ros2 launch rover_bringup rover_bringup.launch.py

# Cleanup on normal exit
cleanup

