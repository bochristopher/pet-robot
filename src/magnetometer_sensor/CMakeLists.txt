cmake_minimum_required(VERSION 3.8)
project(magnetometer_sensor)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(magnetometer_node src/magnetometer_node.cpp)
ament_target_dependencies(magnetometer_node rclcpp sensor_msgs)

install(TARGETS
  magnetometer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
