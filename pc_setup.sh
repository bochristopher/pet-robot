#!/bin/bash
# Run this script on your Ubuntu PC to set up ROS2 for the rover
# Usage: bash pc_setup.sh

echo "=== Rover PC Setup ==="

# Check if ROS2 Humble is installed
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
    echo "ERROR: ROS2 Humble not found at /opt/ros/humble"
    echo "Install ROS2 Humble first: https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

# Add to bashrc if not already present
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
    echo "Added ROS2 source to ~/.bashrc"
fi

if ! grep -q "ROS_DOMAIN_ID=42" ~/.bashrc; then
    echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
    echo "Added ROS_DOMAIN_ID=42 to ~/.bashrc"
fi

# Source for current session
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To visualize the rover, run:"
echo "  source ~/.bashrc"
echo "  ros2 topic list          # Check connection"
echo "  rviz2                    # Open visualizer"
echo ""
echo "Expected topics from Jetson:"
echo "  /scan              - RPLIDAR data"
echo "  /camera/image_raw  - Camera feed"
echo "  /imu/data_raw      - IMU data"
echo "  /ultrasonic/*      - Ultrasonic sensors"
echo "  /cmd_vel           - Motor commands (publish to control)"
echo ""
