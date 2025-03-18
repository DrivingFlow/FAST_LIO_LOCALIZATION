cmake_minimum_required(VERSION 3.8)
project(fast_lio_localization)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fexceptions")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Detect CPU cores for Multi-Processing (MP)
message("Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processor count: ${N}")

  if(N GREATER 4)
    add_definitions(-DMP_EN -DMP_PROC_NUM=3)
    message("MP cores: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN -DMP_PROC_NUM=2)
    message("MP cores: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(dependencies
  rclcpp
  rclpy
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  pcl_ros
  pcl_conversions
  livox_ros_driver2
)

# Messages
set(msg_files
  "msg/Pose6D.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)
ament_export_dependencies(rosidl_default_runtime)

# FastLIO Mapping executable
add_executable(fastlio_mapping
  src/laserMapping.cpp
  include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp
)

target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(fastlio_mapping
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  Python3::Python
)

# Link rosidl typesupport for older ROS 2 distros
list(APPEND EOL_LIST "foxy" "galactic" "eloquent" "dashing" "crystal")
if($ENV{ROS_DISTRO} IN_LIST EOL_LIST)
  rosidl_target_interfaces(fastlio_mapping ${PROJECT_NAME} "rosidl_typesupport_cpp")
else()
  rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(fastlio_mapping ${cpp_typesupport_target})
endif()

ament_target_dependencies(fastlio_mapping ${dependencies})

# ---------------- Install ---------------- #
install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

# ament_python_install_package(${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/global_localization.py
  scripts/publish_initial_pose.py
  scripts/transform_fusion.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()