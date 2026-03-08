import os
from glob import glob
from setuptools import find_packages, setup

package_name = "serial_bridge"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools", "pyserial"],
    zip_safe=True,
    maintainer="bo",
    maintainer_email="bo@todo.todo",
    description="ROS2 serial bridge for Jetson to Arduino Mega communication",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bridge_node = serial_bridge.bridge_node:main",
        ],
    },
)
