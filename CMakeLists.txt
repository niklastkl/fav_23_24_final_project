cmake_minimum_required(VERSION 3.5)
project(final_project)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)


install(PROGRAMS
  nodes/mapper.py
  nodes/path_follower.py
  nodes/path_planner.py
  nodes/position_controller.py
  nodes/scenario_node.py
  nodes/yaw_controller.py
  nodes/robot_marker_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PolygonsStamped.msg"
  "msg/Viewpoint.msg"
  "msg/Viewpoints.msg"
  "srv/SetPath.srv"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs nav_msgs
)

ament_package()
