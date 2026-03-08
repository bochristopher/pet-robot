#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import UnlessCondition


def generate_launch_description():
    pkg_share = get_package_share_directory("rover_bringup")
    default_model = os.path.join(pkg_share, "urdf", "rover_minimal.urdf.xacro")
    default_rviz = os.path.join(pkg_share, "config", "rover.rviz")

    model_arg = DeclareLaunchArgument(
        "model",
        default_value=default_model,
        description="Absolute path to URDF/XACRO model file",
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Start RViz2 with rover config",
    )

    use_joint_state_gui_arg = DeclareLaunchArgument(
        "use_joint_state_gui",
        default_value="false",
        description="Use joint_state_publisher_gui for manual wheel joint motion in RViz",
    )

    robot_description = {
        "robot_description": Command(["xacro ", LaunchConfiguration("model")])
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_gui": False}],
        condition=UnlessCondition(LaunchConfiguration("use_joint_state_gui")),
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_joint_state_gui")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription(
        [
            model_arg,
            use_rviz_arg,
            use_joint_state_gui_arg,
            joint_state_publisher,
            joint_state_publisher_gui,
            robot_state_publisher,
            rviz,
        ]
    )
