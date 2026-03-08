cmake_minimum_required(VERSION 3.16)
project(fast_lio_sam)

SET(CMAKE_BUILD_TYPE "Debug")

ADD_COMPILE_OPTIONS(-std=c++14 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(GTSAM REQUIRED QUIET)

message(Eigen: ${EIGEN3_INCLUDE_DIR})
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
  include)


find_package(catkin QUIET)
if(catkin_FOUND)
  message(STATUS "ROS1 detected")
    set(USE_ROS1 ON)
endif()

find_package(ament_cmake QUIET)
  if(ament_cmake_FOUND)
  message(STATUS "ROS2 detected")
  set(USE_ROS2 ON)
endif()

if(USE_ROS1)
  add_definitions(-DUSE_ROS1)
  find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    sensor_msgs
    roscpp
    rospy
    std_msgs
    pcl_ros
    tf
    livox_ros_driver2
    message_generation
    eigen_conversions
  )

  include_directories(
    ${catkin_INCLUDE_DIRS}
  )

  add_message_files(
    FILES
    Pose6D.msg
  )

  generate_messages(
  DEPENDENCIES
  geometry_msgs
  )

  catkin_package(
    CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
    DEPENDS EIGEN3 PCL GTSAM
    INCLUDE_DIRS
  )

  add_executable(fastlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp src/utility.cpp src/MapOptimization.cpp)
  target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} gtsam)
  target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
endif()

if(USE_ROS2)
  add_definitions(-DUSE_ROS2)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(fast_lio_interfaces REQUIRED)
  find_package(tf2_eigen REQUIRED)
  find_package(livox_ros_driver2)
  find_package(visualization_msgs REQUIRED)
  
  add_executable(fastlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
  ament_target_dependencies(fastlio_mapping
    rclcpp
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
    pcl_conversions
    PCL
    fast_lio_interfaces
    tf2_eigen
    livox_ros_driver2
    visualization_msgs
  )
  target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
  target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
  install(
  DIRECTORY
    launch_ROS2
    config_ROS2
    DESTINATION share/${PROJECT_NAME}
  )
  install(
    FILES 
    rviz_cfg/loam_livox_ros2.rviz
    DESTINATION share/${PROJECT_NAME}/rviz_cfg
  )
  install(TARGETS fastlio_mapping
    DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY include/ DESTINATION include)

  ament_package() # Don't forget this
endif()