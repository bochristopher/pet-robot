#!/bin/bash
# Build ROS2 workspace

cd /home/bo/ros2_ws
source /opt/ros/humble/setup.bash

echo "Building robot_pet package..."
colcon build --packages-select robot_pet --symlink-install

echo ""
echo "Sourcing workspace..."
source install/setup.bash

echo ""
echo "Done! To use:"
echo "  source /home/bo/ros2_ws/install/setup.bash"
echo ""
echo "Launch commands:"
echo "  ros2 launch robot_pet slam.launch.py      # SLAM mapping"
echo "  ros2 launch robot_pet navigation.launch.py map:=/path/to/map.yaml  # Navigation"
