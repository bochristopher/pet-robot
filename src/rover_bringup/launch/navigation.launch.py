from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_share = FindPackageShare("rover_bringup").find("rover_bringup")
    nav2_params = os.path.join(pkg_share, "config", "nav2_params.yaml")
    
    return LaunchDescription([
        # Motor controller (Arduino communication)
        Node(
            package="rover_bringup",
            executable="motor_controller",
            name="motor_controller",
            parameters=[{"serial_port": "/dev/ttyACM0"}],
            output="screen"
        ),
        
        # Camera
        Node(
            package="rover_bringup",
            executable="camera_node",
            name="camera_node",
            parameters=[{
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 15.0,  # Lower FPS for object detection
                "frame_id": "camera_link"
            }],
            output="screen"
        ),
        
        # Object Detection (YOLO)
        Node(
            package="rover_bringup",
            executable="object_detector",
            name="object_detector",
            parameters=[{
                "model": "yolov8n.pt",
                "confidence_threshold": 0.5,
                "publish_annotated": True,
                "obstacle_classes": ["person", "chair", "dog", "cat", "backpack", 
                                     "suitcase", "bottle", "cup", "laptop", "cell phone",
                                     "potted plant", "bed", "dining table", "toilet", 
                                     "tv", "couch", "bicycle", "car", "motorcycle"]
            }],
            output="screen"
        ),
        
        # RPLIDAR
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare("rplidar_ros"), "/launch/rplidar_a1_launch.py"
            ]),
            launch_arguments={
                "serial_port": "/dev/ttyUSB0",
                "frame_id": "laser_frame"
            }.items()
        ),
        
        # Nav2 Navigation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare("nav2_bringup"), "/launch/navigation_launch.py"
            ]),
            launch_arguments={
                "params_file": nav2_params,
                "use_sim_time": "False"
            }.items()
        ),
    ])
