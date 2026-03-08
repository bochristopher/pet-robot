from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'i2c_bus',
            default_value='1',
            description='I2C bus index (/dev/i2c-<bus>)',
        ),
        DeclareLaunchArgument(
            'i2c_address',
            default_value='30',
            description='HMC5883L I2C address in decimal (0x1E = 30)',
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='mag_link',
            description='MagneticField frame_id',
        ),
        DeclareLaunchArgument(
            'publish_rate_hz',
            default_value='50.0',
            description='Publishing rate in Hz',
        ),
        DeclareLaunchArgument(
            'declination_deg',
            default_value='13.5',
            description='Magnetic declination in degrees (east positive)',
        ),
        DeclareLaunchArgument(
            'x_offset',
            default_value='0.0',
            description='Hard-iron X offset in raw sensor units',
        ),
        DeclareLaunchArgument(
            'y_offset',
            default_value='0.0',
            description='Hard-iron Y offset in raw sensor units',
        ),
        DeclareLaunchArgument(
            'z_offset',
            default_value='0.0',
            description='Hard-iron Z offset in raw sensor units',
        ),
        Node(
            package='magnetometer_sensor',
            executable='magnetometer_node',
            name='magnetometer_node',
            output='screen',
            parameters=[{
                'i2c_bus': LaunchConfiguration('i2c_bus'),
                'i2c_address': LaunchConfiguration('i2c_address'),
                'frame_id': LaunchConfiguration('frame_id'),
                'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
                'declination_deg': LaunchConfiguration('declination_deg'),
                'x_offset': LaunchConfiguration('x_offset'),
                'y_offset': LaunchConfiguration('y_offset'),
                'z_offset': LaunchConfiguration('z_offset'),
            }],
        ),
    ])
