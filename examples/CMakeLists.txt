find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT ipopt QUIET)

add_executable(asif_doubleintegrator asif_doubleintegrator.cpp)
target_link_libraries(asif_doubleintegrator feedback)
target_compile_options(asif_doubleintegrator PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(mpc_doubleintegrator mpc_doubleintegrator.cpp)
target_link_libraries(mpc_doubleintegrator feedback)
target_compile_options(mpc_doubleintegrator PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(mpc_asif_vehicle mpc_asif_vehicle.cpp)
target_link_libraries(mpc_asif_vehicle feedback)
target_compile_options(mpc_asif_vehicle PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(pid_se2 pid_se2.cpp)
target_link_libraries(pid_se2 feedback)
target_compile_options(pid_se2 PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(snippets snippets.cpp)
target_link_libraries(snippets feedback)

if (${IPOPT_FOUND})
  add_executable(colloc_doubleintegrator colloc_doubleintegrator.cpp)
  target_include_directories(colloc_doubleintegrator SYSTEM PRIVATE ${IPOPT_INCLUDE_DIRS})
  target_link_libraries(colloc_doubleintegrator feedback ${IPOPT_LIBRARIES})
  target_compile_options(colloc_doubleintegrator PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

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

  if (${IPOPT_FOUND})
    target_compile_definitions(colloc_doubleintegrator PRIVATE ENABLE_PLOTTING)
    target_link_libraries(colloc_doubleintegrator Matplot++::matplot)
  endif()
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

