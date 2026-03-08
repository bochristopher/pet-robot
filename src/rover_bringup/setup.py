from setuptools import setup
import os
from glob import glob

package_name = "rover_bringup"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config"), glob("config/*.rviz")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bo",
    maintainer_email="bo@rover.local",
    description="Rover bringup package",
    license="MIT",
    entry_points={
        "console_scripts": [
            "imu_node = rover_bringup.imu_node:main",
            "camera_node = rover_bringup.camera_node:main",
            "oak_camera_node = rover_bringup.oak_camera_node:main",
            "object_detector = rover_bringup.object_detector:main",
            "lidar_filter = rover_bringup.lidar_filter:main",
            "odometry_node = rover_bringup.odometry_node:main",
            "battery_monitor = rover_bringup.battery_monitor:main",
            "police_mode_node = rover_bringup.police_mode_node:main",
            "wanderer = rover_bringup.wanderer:main",
        ],
    },
)
