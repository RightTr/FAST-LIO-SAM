cmake_minimum_required(VERSION 3.16)
project(livox_ros_driver2)

if(DEFINED FASTLIO_ROS_VERSION AND NOT "${FASTLIO_ROS_VERSION}" STREQUAL "")
  set(_ros_version "${FASTLIO_ROS_VERSION}")
else()
  set(_ros_version "$ENV{ROS_VERSION}")
endif()

if(_ros_version MATCHES "^(1|ROS1|ros1)$")
  find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)
  add_message_files(FILES CustomPoint.msg CustomMsg.msg)
  generate_messages(DEPENDENCIES std_msgs)
  catkin_package(CATKIN_DEPENDS message_runtime std_msgs)
elseif(_ros_version MATCHES "^(2|ROS2|ros2)$")
  find_package(ament_cmake REQUIRED)
  find_package(rosidl_default_generators REQUIRED)
  find_package(std_msgs REQUIRED)
  rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/CustomPoint.msg"
    "msg/CustomMsg.msg"
    DEPENDENCIES std_msgs
  )
  ament_export_dependencies(rosidl_default_runtime)
  ament_package()
else()
  message(FATAL_ERROR "FASTLIO_ROS_VERSION must be 1 or 2")
endif()
