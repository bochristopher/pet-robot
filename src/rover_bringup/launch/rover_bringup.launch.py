#!/usr/bin/env python3
"""
Rover Bringup Launch File

Properly manages all robot nodes with correct lifecycle handling.
This prevents ROS2 context corruption that occurs with bash scripts.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_slam = DeclareLaunchArgument(
        'use_slam', default_value='false',
        description='Start SLAM Toolbox'
    )
    
    serial_port = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyACM0',
        description='micro-ROS serial device'
    )

    micro_ros_baudrate = DeclareLaunchArgument(
        'micro_ros_baudrate', default_value='115200',
        description='micro-ROS serial baudrate'
    )
    
    lidar_port = DeclareLaunchArgument(
        'lidar_port', default_value='/dev/ttyUSB0',
        description='LIDAR serial port'
    )

    # micro-ROS Agent (bridges DDS <-> USB serial for Arduino node)
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
        output='screen'
    )

    # RPLidar
    rplidar = TimerAction(
        period=3.0,
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
                output='screen'
            )
        ]
    )

    # IMU Node
    imu_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='rover_bringup',
                executable='imu_node',
                name='imu_node',
                parameters=[{'i2c_bus': 7}],
                output='screen'
            )
        ]
    )

    # Odometry Node
    odometry_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='rover_bringup',
                executable='odometry_node',
                name='odometry_node',
                parameters=[{
                    'wheel_radius': 0.0381,
                    'wheel_base': 0.2286,
                    'publish_tf': True,
                    'use_imu': True
                }],
                output='screen'
            )
        ]
    )

    # Foxglove Bridge
    foxglove_bridge = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='foxglove_bridge',
                executable='foxglove_bridge',
                name='foxglove_bridge',
                output='screen'
            )
        ]
    )

    # Static TF: base_link -> laser (LIDAR rotated 180°)
    tf_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_laser',
        arguments=['0', '0', '0.15', '3.14159', '0', '0', 'base_link', 'laser']
    )

    # Static TF: base_link -> camera_link
    tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_camera',
        arguments=['0.1', '0', '0.05', '0', '0', '0', 'base_link', 'camera_link']
    )

    # Static TF: Ultrasonic sensors
    tf_us_fl = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_ultrasonic_fl',
        arguments=['0.1', '0.1', '0.05', '0', '0', '0.3', 'base_link', 'ultrasonic_front_left_link']
    )

    tf_us_fr = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_ultrasonic_fr',
        arguments=['0.1', '-0.1', '0.05', '0', '0', '-0.3', 'base_link', 'ultrasonic_front_right_link']
    )

    tf_us_bc = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_ultrasonic_bc',
        arguments=['-0.1', '0', '0.05', '0', '0', '3.14159', 'base_link', 'ultrasonic_back_center_link']
    )

    # Log startup info
    startup_info = LogInfo(
        msg='\n==========================================\n'
            '  Pet Robot - ROS2 Bringup (Launch File)\n'
            '==========================================\n'
            '  Use Ctrl+C to cleanly stop all nodes\n'
    )

    return LaunchDescription([
        # Launch arguments
        use_slam,
        serial_port,
        micro_ros_baudrate,
        lidar_port,
        
        # Startup message
        startup_info,
        
        # Static transforms (start immediately)
        tf_laser,
        tf_camera,
        tf_us_fl,
        tf_us_fr,
        tf_us_bc,
        
        # Core nodes with staggered startup
        micro_ros_agent,
        rplidar,
        imu_node,
        odometry_node,
        foxglove_bridge,
    ])

