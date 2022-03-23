include(GoogleTest)
find_package(GTest REQUIRED)

configure_file(
  ${CMAKE_ROOT}/Modules/DartConfiguration.tcl.in
  ${PROJECT_BINARY_DIR}/DartConfiguration.tcl
)

add_library(TestConfig INTERFACE)
target_compile_options(TestConfig
INTERFACE
  -Wall
  -Wextra
  -Wshadow
  -Wnon-virtual-dtor
  -Wold-style-cast
  -Wcast-align
  -Wunused
  -Woverloaded-virtual
  -Wpedantic
  # -Wconversion        TODO enable and fix
  # -Wsign-conversion   TODO enable and fix
  -Wnull-dereference
  # -Wdouble-promotion  TODO enable and fix
  -Wimplicit-fallthrough
  -Werror
)
target_link_libraries(TestConfig
INTERFACE
  feedback
  GTest::Main
  -fsanitize=undefined,address
)

add_executable(test_ocp_to_qp test_ocp_to_qp.cpp)
target_link_libraries(test_ocp_to_qp PRIVATE TestConfig)
gtest_discover_tests(test_ocp_to_qp)

add_executable(test_static test_static.cpp)
target_link_libraries(test_static PRIVATE TestConfig)
gtest_discover_tests(test_static)

find_package(rclcpp QUIET)
if (${rclcpp_FOUND})
  target_link_libraries(test_static PRIVATE rclcpp::rclcpp)
  target_compile_definitions(test_static PRIVATE ENABLE_ROS_TESTS)
endif()

add_executable(test_asif test_asif.cpp)
target_link_libraries(test_asif PRIVATE TestConfig)
gtest_discover_tests(test_asif)

add_executable(test_ekf test_ekf.cpp)
target_link_libraries(test_ekf PRIVATE TestConfig)
gtest_discover_tests(test_ekf)

add_executable(test_mpc test_mpc.cpp)
target_link_libraries(test_mpc PRIVATE TestConfig)
gtest_discover_tests(test_mpc)

add_executable(test_qp test_qp.cpp)
target_link_libraries(test_qp PRIVATE TestConfig)
gtest_discover_tests(test_qp)

add_executable(test_pid test_pid.cpp)
target_link_libraries(test_pid PRIVATE TestConfig)
gtest_discover_tests(test_pid)

add_executable(test_collocation_mesh test_collocation_mesh.cpp)
target_link_libraries(test_collocation_mesh PRIVATE TestConfig)
gtest_discover_tests(test_collocation_mesh)

add_executable(test_collocation_mesh_function test_collocation_mesh_function.cpp)
target_link_libraries(test_collocation_mesh_function PRIVATE TestConfig)
gtest_discover_tests(test_collocation_mesh_function)

add_executable(test_collocation_dyn_error test_collocation_dyn_error.cpp)
target_link_libraries(test_collocation_dyn_error PRIVATE TestConfig)
gtest_discover_tests(test_collocation_dyn_error)

add_executable(test_utils_sparse test_utils_sparse.cpp)
target_link_libraries(test_utils_sparse PRIVATE TestConfig)
gtest_discover_tests(test_utils_sparse)

add_executable(test_ocp_to_nlp test_ocp_to_nlp.cpp)
target_link_libraries(test_ocp_to_nlp PRIVATE TestConfig)
gtest_discover_tests(test_ocp_to_nlp)

find_package(autodiff 0.6 QUIET)
if (autodiff_FOUND)
  add_executable(test_ocp_flatten test_ocp_flatten.cpp)
  target_link_libraries(test_ocp_flatten PRIVATE TestConfig autodiff::autodiff)
  gtest_discover_tests(test_ocp_flatten)
endif()

pkg_check_modules(IPOPT ipopt QUIET)
if (${IPOPT_FOUND})
  add_executable(test_ocp_ipopt test_ocp_ipopt.cpp)
  target_include_directories(test_ocp_ipopt SYSTEM PRIVATE ${IPOPT_INCLUDE_DIRS})
  target_link_libraries(test_ocp_ipopt PRIVATE TestConfig ${IPOPT_LIBRARIES})
  gtest_discover_tests(test_ocp_ipopt)
endif()
