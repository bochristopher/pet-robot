#!/usr/bin/env python3
"""Autonomous exploration launch file - SLAM + Nav2 + Explorer."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('robot_pet')

    slam_config = os.path.join(pkg_dir, 'config', 'slam_toolbox.yaml')
    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')

    # URDF file
    urdf_file = os.path.join(pkg_dir, 'urdf', 'robot_pet.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
        ),

        # Arduino Bridge
        Node(
            package='robot_pet',
            executable='arduino_bridge',
            name='arduino_bridge',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyACM0',
                'baud_rate': 115200,
            }]
        ),

        # Odometry Node
        Node(
            package='robot_pet',
            executable='odometry_node',
            name='odometry_node',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyACM0',
                'baud_rate': 115200,
                'publish_tf': True,
            }]
        ),

        # RPLidar
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': 115200,
                'frame_id': 'laser',
                'angle_compensate': True,
            }]
        ),

        # SLAM Toolbox (provides map and localization)
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_config],
        ),

        # Nav2 Controller Server (local path following)
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[nav2_params],
        ),

        # Nav2 Planner Server (global path planning)
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[nav2_params],
        ),

        # Nav2 Behavior Server (recovery behaviors)
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[nav2_params],
        ),

        # Nav2 BT Navigator (behavior tree navigation)
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[nav2_params],
        ),

        # Nav2 Lifecycle Manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': [
                    'controller_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator',
                ]
            }]
        ),

        # Autonomous Explorer (delayed start to let Nav2 initialize)
        TimerAction(
            period=10.0,  # Wait 10 seconds for Nav2 to start
            actions=[
                Node(
                    package='robot_pet',
                    executable='autonomous_explorer',
                    name='autonomous_explorer',
                    output='screen',
                    parameters=[{
                        'min_frontier_size': 10,
                        'goal_tolerance': 0.5,
                        'exploration_timeout': 300.0,
                        'benchmark_mode': True,
                    }]
                ),
            ]
        ),

        # Camera with obstacle detection
        Node(
            package='robot_pet',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[{
                'device': '/dev/video0',
                'width': 640,
                'height': 480,
                'fps': 10,
                'enable_depth': False,  # Set True if you have torch/MiDaS
            }]
        ),

        # RViz disabled - no display available
        # To enable, run separately: rviz2
    ])
