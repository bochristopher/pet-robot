from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # micro-ROS agent
        Node(
            package="micro_ros_agent",
            executable="micro_ros_agent",
            name="micro_ros_agent",
            arguments=["serial", "--dev", "/dev/ttyACM0", "--baudrate", "115200"],
            output="screen"
        ),
        
        # RPLIDAR
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare("rplidar_ros"), "/launch/rplidar_a1_launch.py"
            ]),
            launch_arguments={
                "serial_port": "/dev/ttyUSB0",
                "frame_id": "laser"
            }.items()
        ),
        
        # Static TF: base_link -> laser
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "0", "0.1", "0", "0", "0", "base_link", "laser"]
        ),
        
        # Static TF: odom -> base_link (temporary - should be from odometry)
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "odom", "base_link"]
        ),
        
        # Static TF: map -> odom
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"]
        ),
    ])
