"""Launch the serial_bridge node with YAML parameters."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_dir = get_package_share_directory("serial_bridge")
    default_params = os.path.join(pkg_dir, "config", "bridge_params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to the bridge parameters YAML file",
        ),
        DeclareLaunchArgument(
            "serial_port",
            default_value="/dev/ttyACM0",
            description="Serial port for Arduino Mega",
        ),
        Node(
            package="serial_bridge",
            executable="bridge_node",
            name="serial_bridge",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {"serial_port": LaunchConfiguration("serial_port")},
            ],
        ),
    ])
