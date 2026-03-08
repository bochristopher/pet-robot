#!/usr/bin/env bash
set -eo pipefail

# Stop launchers first
pkill -f ros2 launch rover_bringup rover_bringup.launch.py || true
pkill -f ros2 launch rover_bringup full_bringup.launch.py || true
pkill -f ros2 launch linorobot2_navigation slam.launch.py || true

# Stop child nodes that frequently get orphaned
pkill -f /rover_bringup/lib/rover_bringup/ || true
pkill -f /rplidar_ros/rplidar_node || true
pkill -f /micro_ros_agent/micro_ros_agent || true
pkill -f /foxglove_bridge/foxglove_bridge || true
pkill -f /slam_toolbox/async_slam_toolbox_node || true
pkill -f /nav2_controller/controller_server || true
pkill -f /nav2_planner/planner_server || true
pkill -f /nav2_behaviors/behavior_server || true
pkill -f /nav2_bt_navigator/bt_navigator || true
pkill -f /nav2_waypoint_follower/waypoint_follower || true
pkill -f /nav2_velocity_smoother/velocity_smoother || true
pkill -f /nav2_lifecycle_manager/lifecycle_manager || true
pkill -f teleop_twist_keyboard || true
pkill -f static_transform_publisher || true

sleep 2
echo Rover processes stopped.
