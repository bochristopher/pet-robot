#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/bo/rover_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/fastdds_robot.xml
export LINOROBOT2_BASE=4wd

# Ensure core bringup is active first
if ! pgrep -f "ros2 launch rover_bringup rover_bringup.launch.py" >/dev/null; then
  nohup ros2 launch rover_bringup rover_bringup.launch.py > /tmp/rover_bringup.log 2>&1 &
  sleep 8
fi

# Stop previous SLAM/Nav2 stack only
pkill -f "ros2 launch linorobot2_navigation slam.launch.py" || true
pkill -f "/nav2_controller/controller_server" || true
pkill -f "/nav2_planner/planner_server" || true
pkill -f "/nav2_behaviors/behavior_server" || true
pkill -f "/nav2_bt_navigator/bt_navigator" || true
pkill -f "/nav2_waypoint_follower/waypoint_follower" || true
pkill -f "/nav2_velocity_smoother/velocity_smoother" || true
pkill -f "/nav2_lifecycle_manager/lifecycle_manager" || true
pkill -f "/slam_toolbox/async_slam_toolbox_node" || true
sleep 2

nohup ros2 launch linorobot2_navigation slam.launch.py > /tmp/linorobot2_slam.log 2>&1 &
echo "Started linorobot2 SLAM"
echo "Log: /tmp/linorobot2_slam.log"
