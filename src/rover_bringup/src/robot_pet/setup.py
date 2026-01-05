import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robot_pet'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        # URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        # Maps directory
        (os.path.join('share', package_name, 'maps'), []),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='bo',
    maintainer_email='bochristopher.engineer@protonmail.com',
    description='ROS2 package for robot pet with Nav2 and SLAM',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'arduino_bridge = robot_pet.arduino_bridge:main',
            'odometry_node = robot_pet.odometry_node:main',
            'autonomous_explorer = robot_pet.autonomous_explorer:main',
            'camera_node = robot_pet.camera_node:main',
        ],
    },
)
