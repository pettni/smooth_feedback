include(GoogleTest)
find_package(GTest REQUIRED)

configure_file(
  ${CMAKE_ROOT}/Modules/DartConfiguration.tcl.in
  ${PROJECT_BINARY_DIR}/DartConfiguration.tcl
)

add_executable(test_ocp_to_qp test_ocp_to_qp.cpp)
target_link_libraries(test_ocp_to_qp PRIVATE feedback GTest::Main)
gtest_discover_tests(test_ocp_to_qp)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_executable(test_static test_static.cpp)
target_link_libraries(test_static PRIVATE feedback GTest::Main)
gtest_discover_tests(test_static)

find_package(rclcpp QUIET)
if (${rclcpp_FOUND})
  target_link_libraries(test_static PRIVATE rclcpp::rclcpp)
  target_compile_definitions(test_static PRIVATE ENABLE_ROS_TESTS)
endif()

add_executable(test_asif test_asif.cpp)
target_link_libraries(test_asif PRIVATE feedback GTest::Main)
gtest_discover_tests(test_asif)

add_executable(test_ekf test_ekf.cpp)
target_link_libraries(test_ekf PRIVATE feedback GTest::Main)
gtest_discover_tests(test_ekf)

add_executable(test_mpc test_mpc.cpp)
target_link_libraries(test_mpc PRIVATE feedback GTest::Main)
gtest_discover_tests(test_mpc)

add_executable(test_qp test_qp.cpp)
target_link_libraries(test_qp PRIVATE feedback GTest::Main)
gtest_discover_tests(test_qp)

add_executable(test_pid test_pid.cpp)
target_link_libraries(test_pid PRIVATE feedback GTest::Main)
gtest_discover_tests(test_pid)

add_executable(test_collocation test_collocation.cpp)
target_link_libraries(test_collocation PRIVATE feedback GTest::Main)
gtest_discover_tests(test_collocation)

add_executable(test_collocation_eval_reduce test_collocation_eval_reduce.cpp)
target_link_libraries(test_collocation_eval_reduce PRIVATE feedback GTest::Main)
gtest_discover_tests(test_collocation_eval_reduce)

add_executable(test_utils_sparse test_utils_sparse.cpp)
target_link_libraries(test_utils_sparse PRIVATE feedback GTest::Main)
gtest_discover_tests(test_utils_sparse)

add_executable(test_ocp test_ocp.cpp)
target_link_libraries(test_ocp PRIVATE feedback GTest::Main)
gtest_discover_tests(test_ocp)

pkg_check_modules(IPOPT ipopt QUIET)
if (${IPOPT_FOUND})
  add_executable(test_ocp_ipopt test_ocp_ipopt.cpp)
  target_include_directories(test_ocp_ipopt SYSTEM PRIVATE ${IPOPT_INCLUDE_DIRS})
  target_link_libraries(test_ocp_ipopt PRIVATE feedback ${IPOPT_LIBRARIES} GTest::Main)
  gtest_discover_tests(test_ocp_ipopt)
endif()
