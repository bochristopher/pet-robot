#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/bo/rover_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/fastdds_robot.xml
export LINOROBOT2_BASE=4wd

/home/bo/rover_ws/scripts/rover_stop.sh >/dev/null 2>&1 || true

nohup ros2 launch rover_bringup rover_bringup.launch.py > /tmp/rover_bringup.log 2>&1 &
echo "Started rover_bringup.launch.py"
echo "Log: /tmp/rover_bringup.log"
