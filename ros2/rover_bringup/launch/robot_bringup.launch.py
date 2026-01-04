#!/usr/bin/env python3
"""
Robot Bringup Launch File
Launches all hardware interface nodes for the rover.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch arguments
    use_rplidar = LaunchConfiguration('use_rplidar', default='true')
    use_camera = LaunchConfiguration('use_camera', default='true')
    use_imu = LaunchConfiguration('use_imu', default='true')
    use_motors = LaunchConfiguration('use_motors', default='true')

    # RPLIDAR node (use /dev/rplidar after udev rules installed, else /dev/ttyUSB0)
    rplidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': 115200,  # A1/A2: 115200, A3: 256000
            'frame_id': 'laser_frame',
            'angle_compensate': True,
            'scan_mode': 'Sensitivity',
        }],
        output='screen',
    )

    # Camera node (OpenCV-based, works better on Jetson)
    camera_node = Node(
        package='rover_bringup',
        executable='camera_node',
        name='camera_node',
        parameters=[{
            'device_id': 0,
            'width': 640,
            'height': 480,
            'fps': 30.0,
            'frame_id': 'camera_link',
            'publish_compressed': True,
        }],
        output='screen',
    )

    # IMU node (MPU6050 on Jetson 40-pin header I2C = bus 7)
    imu_node = Node(
        package='rover_bringup',
        executable='imu_node',
        name='imu_node',
        parameters=[{
            'i2c_bus': 7,
            'i2c_address': 0x68,
            'frame_id': 'imu_link',
            'publish_rate': 100.0,
        }],
        output='screen',
    )

    # Motor controller node (also handles ultrasonic sensors via Arduino)
    motor_node = Node(
        package='rover_bringup',
        executable='motor_controller',
        name='motor_controller',
        parameters=[{
            'serial_port': '/dev/ttyACM0',  # Use /dev/arduino after udev rules
            'baud_rate': 115200,
            'wheel_base': 0.3,
            'wheel_radius': 0.05,
            'max_speed': 1.0,
            'ultrasonic_sensors': ['front_left', 'front_right', 'back_center'],
        }],
        output='screen',
    )

    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument('use_rplidar', default_value='true'),
        DeclareLaunchArgument('use_camera', default_value='true'),
        DeclareLaunchArgument('use_imu', default_value='true'),
        DeclareLaunchArgument('use_motors', default_value='true'),

        # Launch nodes
        rplidar_node,
        camera_node,
        imu_node,
        motor_node,
    ])
