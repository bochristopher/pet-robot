#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/bo/rover_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "== Launchers =="
ps -eo pid,args | sed -n '/ros2 launch rover_bringup rover_bringup.launch.py/p;/ros2 launch linorobot2_navigation slam.launch.py/p'

echo ""
echo "== Key Nodes =="
ros2 node list | grep -E "micro_ros_agent|rplidar_node|imu_node|odometry_node|slam_toolbox" || true

echo ""
echo "== Key Topics =="
ros2 topic list | grep -E "^/scan$|^/odom$|^/imu/data$|^/map$|^/tf$" || true
