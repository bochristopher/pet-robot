#!/bin/bash
# ==========================================
#  Pet Robot - One Command Startup
# ==========================================
# Usage: ./start.sh
#
# This script:
# 1. Cleans up any orphaned processes
# 2. Resets ROS2 daemon
# 3. Launches everything with respawn enabled
#
# Press Ctrl+C to cleanly stop all nodes

set -e

export ROS_DOMAIN_ID=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "=========================================="
echo "  Pet Robot - Starting Up"
echo "=========================================="
echo -e "${NC}"

# Step 1: Clean up
echo -e "${YELLOW}[1/4] Cleaning up previous session...${NC}"
pkill -9 -f "micro_ros_agent|rplidar|lidar_filter|imu_node|odometry|foxglove|oak_camera|slam_toolbox|static_transform" 2>/dev/null || true
sleep 1

# Step 2: Reset ROS2
echo -e "${YELLOW}[2/4] Resetting ROS2 daemon...${NC}"
ros2 daemon stop 2>/dev/null || true
rm -rf /dev/shm/fastrtps* /dev/shm/ros2* 2>/dev/null || true
sleep 1
ros2 daemon start
sleep 1

# Step 3: Source environment
echo -e "${YELLOW}[3/4] Setting up environment...${NC}"
source /opt/ros/humble/setup.bash
source ~/rover_ws/install/setup.bash

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo -e "${YELLOW}[4/4] Launching all nodes...${NC}"
echo ""
echo -e "${GREEN}=========================================="
echo "  Foxglove: ws://${LOCAL_IP}:8765"
echo "  Nodes will auto-restart if they crash"
echo "  Press Ctrl+C to stop everything"
echo -e "==========================================${NC}"
echo ""

# Launch everything
ros2 launch rover_bringup full_bringup.launch.py

echo ""
echo -e "${RED}All nodes stopped.${NC}"

