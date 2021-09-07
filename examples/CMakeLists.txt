add_executable(asif_doubleintegrator asif_doubleintegrator.cpp)
target_link_libraries(asif_doubleintegrator feedback)

add_executable(mpc_doubleintegrator mpc_doubleintegrator.cpp)
target_link_libraries(mpc_doubleintegrator feedback)

add_executable(mpc_asif_vehicle mpc_asif_vehicle.cpp)
target_link_libraries(mpc_asif_vehicle feedback)

add_executable(pid_se2 pid_se2.cpp)
target_link_libraries(pid_se2 feedback)

add_executable(snippets snippets.cpp)
target_link_libraries(snippets feedback)

find_package(Matplot++ QUIET)
if(${Matplot++_FOUND})
  set(CMAKE_CXX_STANDARD 20)  # thanks Matplot++
  target_compile_definitions(asif_doubleintegrator PRIVATE ENABLE_PLOTTING)
  target_link_libraries(asif_doubleintegrator Matplot++::matplot)

  target_compile_definitions(mpc_doubleintegrator PRIVATE ENABLE_PLOTTING)
  target_link_libraries(mpc_doubleintegrator Matplot++::matplot)

  target_compile_definitions(mpc_asif_vehicle PRIVATE ENABLE_PLOTTING)
  target_link_libraries(mpc_asif_vehicle Matplot++::matplot)

  target_compile_definitions(pid_se2 PRIVATE ENABLE_PLOTTING)
  target_link_libraries(pid_se2 Matplot++::matplot)
else()
  message(WARNING "Matplot++ not found, plotting disabled")
endif()

find_package(rclcpp QUIET)
find_package(gazebo_msgs QUIET)
find_package(geometry_msgs QUIET)
if (${rclcpp_FOUND} AND ${gazebo_msgs_FOUND} AND ${geometry_msgs_FOUND})
  target_compile_definitions(mpc_asif_vehicle PRIVATE ENABLE_ROS)
  target_link_libraries(mpc_asif_vehicle rclcpp::rclcpp)
  target_link_libraries(mpc_asif_vehicle gazebo_msgs::gazebo_msgs__rosidl_typesupport_cpp)
  target_link_libraries(mpc_asif_vehicle geometry_msgs::geometry_msgs__rosidl_typesupport_cpp)
else()
  message(WARNING "ROS parts not found")
endif()
