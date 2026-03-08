#!/usr/bin/env python3
"""
Police Mode Launch File - NO MICROPHONE
=======================================
Speech output only. No audio input/listening.

Usage:
  ros2 launch rover_bringup police_mode.launch.py
  ros2 launch rover_bringup police_mode.launch.py voice:=adam
  ros2 launch rover_bringup police_mode.launch.py tracking:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    
    tracking_arg = DeclareLaunchArgument('tracking', default_value='true')
    voice_arg = DeclareLaunchArgument('voice', default_value='adam')
    confidence_arg = DeclareLaunchArgument('confidence', default_value='0.5')
    
    oak_camera_node = Node(
        package='rover_bringup',
        executable='oak_camera_node',
        name='oak_camera_node',
        output='screen',
        parameters=[{
            'publish_rgb': True,
            'publish_depth_image': False,
            'flip_image': True,
        }],
        remappings=[('/camera/rgb/image_raw', '/camera/image_raw')]
    )
    
    object_detector_node = Node(
        package='rover_bringup',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[{
            'model': 'yolov8n.pt',
            'confidence_threshold': LaunchConfiguration('confidence'),
            'publish_annotated': True,
            'obstacle_classes': ['person'],
        }]
    )
    
    police_mode_node = Node(
        package='rover_bringup',
        executable='police_mode_node',
        name='police_mode_node',
        output='screen',
        parameters=[{
            'enabled': True,
            'elevenlabs_voice': LaunchConfiguration('voice'),
            'use_elevenlabs': True,
            'cache_audio': True,
            'tracking_enabled': LaunchConfiguration('tracking'),
            'tracking_speed': 0.15,
            'tracking_angular': 0.8,
            'confidence_threshold': LaunchConfiguration('confidence'),
            'initial_cooldown': 8.0,
            'followup_cooldown': 6.0,
        }]
    )
    
    return LaunchDescription([
        tracking_arg,
        voice_arg,
        confidence_arg,
        oak_camera_node,
        object_detector_node,
        police_mode_node,
    ])
