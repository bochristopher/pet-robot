#!/usr/bin/env python3
"""
Full Rover Bringup Launch File

Starts ALL nodes including SLAM in a single, stable launch.
Uses proper lifecycle management to prevent ROS2 context corruption.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package path
    pkg_path = get_package_share_directory('rover_bringup')
    slam_params = os.path.join(pkg_path, 'config', 'slam_params.yaml')
    
    # Launch arguments
    use_slam_arg = DeclareLaunchArgument(
        'use_slam', default_value='true',
        description='Start SLAM Toolbox'
    )
    
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyACM0',
        description='micro-ROS serial device'
    )

    micro_ros_baudrate_arg = DeclareLaunchArgument(
        'micro_ros_baudrate', default_value='115200',
        description='micro-ROS serial baudrate'
    )
    
    lidar_port_arg = DeclareLaunchArgument(
        'lidar_port', default_value='/dev/ttyUSB0',
        description='LIDAR serial port'
    )

    # ============ STATIC TRANSFORMS (Start first) ============
    
    # base_link -> laser (LIDAR rotated 180°)
    tf_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_laser',
        arguments=['--x', '0', '--y', '0', '--z', '0.1',
                   '--roll', '3.14159', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'laser']
    )

    # base_link -> camera_link
    tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_camera',
        arguments=['--x', '0.1', '--y', '0', '--z', '0.05',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link']
    )

    # Ultrasonic sensors
    tf_us_fl = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_ultrasonic_fl',
        arguments=['--x', '0.1', '--y', '0.1', '--z', '0.05',
                   '--roll', '0', '--pitch', '0', '--yaw', '0.3',
                   '--frame-id', 'base_link', '--child-frame-id', 'ultrasonic_front_left_link']
    )

    tf_us_fr = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_ultrasonic_fr',
        arguments=['--x', '0.1', '--y', '-0.1', '--z', '0.05',
                   '--roll', '0', '--pitch', '0', '--yaw', '-0.3',
                   '--frame-id', 'base_link', '--child-frame-id', 'ultrasonic_front_right_link']
    )

    tf_us_bc = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_ultrasonic_bc',
        arguments=['--x', '-0.1', '--y', '0', '--z', '0.05',
                   '--roll', '0', '--pitch', '0', '--yaw', '3.14159',
                   '--frame-id', 'base_link', '--child-frame-id', 'ultrasonic_back_center_link']
    )

    # ============ CORE NODES ============
    
    # micro-ROS Agent (starts immediately)
    micro_ros_agent = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=[
            'serial',
            '--dev',
            LaunchConfiguration('serial_port'),
            '--baudrate',
            LaunchConfiguration('micro_ros_baudrate'),
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0
    )

    # RPLidar (delay 2s)
    rplidar = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='rplidar_ros',
                executable='rplidar_node',
                name='rplidar_node',
                parameters=[{
                    'serial_port': LaunchConfiguration('lidar_port'),
                    'serial_baudrate': 115200,
                    'frame_id': 'laser'
                }],
                output='screen',
                respawn=True,
                respawn_delay=3.0
            )
        ]
    )

    # IMU Node (delay 3s)
    imu_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='rover_bringup',
                executable='imu_node',
                name='imu_node',
                parameters=[{'i2c_bus': 7}],
                output='screen',
                respawn=True,
                respawn_delay=2.0
            )
        ]
    )

    # Odometry Node (delay 4s)
    odometry_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='rover_bringup',
                executable='odometry_node',
                name='odometry_node',
                parameters=[{
                    'wheel_radius': 0.0381,
                    'wheel_base': 0.2286,
                    'publish_tf': True,
                    'use_imu': True,
                    'imu_alpha': 0.98
                }],
                output='screen',
                respawn=True,
                respawn_delay=2.0
            )
        ]
    )

    # OAK-D Camera (delay 3s)
    oak_camera = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='rover_bringup',
                executable='oak_camera_node',
                name='oak_depth_node',
                parameters=[{
                    # Improve depth coverage while keeping performance reasonable
                    'point_cloud_decimation': 2,
                    'max_depth_mm': 6000,
                    'min_depth_mm': 200,
                    'publish_depth_image': False,
                    'publish_rgb': True,
                }],
                output='screen',
                respawn=True,
                respawn_delay=3.0
            )
        ]
    )

    # Foxglove Bridge (delay 5s)
    foxglove_bridge = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='foxglove_bridge',
                executable='foxglove_bridge',
                name='foxglove_bridge',
                output='screen',
                respawn=True,
                respawn_delay=2.0
            )
        ]
    )

    # ============ SLAM TOOLBOX (delay 6s) ============
    slam_toolbox = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                parameters=[slam_params],
                output='screen',
                respawn=True,
                respawn_delay=5.0
            )
        ]
    )

    # ============ STARTUP INFO ============
    startup_info = LogInfo(
        msg='\n'
            '==========================================\n'
            '  Pet Robot - Full Bringup\n'
            '==========================================\n'
            '  All nodes starting with respawn enabled\n'
            '  Press Ctrl+C to cleanly stop all nodes\n'
            '==========================================\n'
    )

    return LaunchDescription([
        # Arguments
        use_slam_arg,
        serial_port_arg,
        micro_ros_baudrate_arg,
        lidar_port_arg,
        
        # Info
        startup_info,
        
        # Static transforms first (immediate)
        tf_laser,
        tf_camera,
        tf_us_fl,
        tf_us_fr,
        tf_us_bc,
        
        # Core nodes with staggered startup and respawn
        micro_ros_agent,
        rplidar,
        imu_node,
        oak_camera,
        odometry_node,
        foxglove_bridge,
        slam_toolbox,
    ])

