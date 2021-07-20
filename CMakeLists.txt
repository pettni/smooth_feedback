cmake_minimum_required(VERSION 3.15)
project(smooth_feedback VERSION 0.0.1)

include(CTest)
include(GoogleTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_EXAMPLES "Build the examples." OFF)

find_package(Boost REQUIRED)
find_package(smooth REQUIRED)

# ---------------------------------------------------------------------------------------
# TARGETS
# ---------------------------------------------------------------------------------------

add_library(feedback INTERFACE)
target_include_directories(feedback
INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(feedback
INTERFACE
  Boost::headers
  smooth::smooth
)

# ---------------------------------------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------------------------------------

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------------------
# INSTALLATION
# ---------------------------------------------------------------------------------------

include(GNUInstallDirs)

# Header files
install(
  DIRECTORY    ${PROJECT_SOURCE_DIR}/include/smooth
  DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}
)

# Targets
install(
  TARGETS               feedback
  EXPORT                ${PROJECT_NAME}_targets
  LIBRARY DESTINATION   ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION   ${CMAKE_INSTALL_LIBDIR}
)

# ---------------------------------------------------------------------------------------
# CMAKE CONFIGURATION
# ---------------------------------------------------------------------------------------

set(export_dest_dir ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Generate TARGETS files
install(
  EXPORT       ${PROJECT_NAME}_targets
  FILE         ${PROJECT_NAME}Targets.cmake
  NAMESPACE    smooth::
  DESTINATION  ${export_dest_dir}
)

# Generate CONFIG file
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/ConfigTemplate.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${export_dest_dir}
)

# Generate VERSION file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Version.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install CONFIG and VERSION files
install(
  FILES        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Version.cmake
  DESTINATION  ${export_dest_dir}
)

# ---------------------------------------------------------------------------------------
# TESTING
# ---------------------------------------------------------------------------------------

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  find_package(autodiff REQUIRED)
  find_package(Ceres REQUIRED)

  add_compile_options(-Wall -Wextra -Wpedantic -Werror)

  # add_executable(test_asif test/test_asif.cpp)
  # target_link_libraries(test_asif PRIVATE feedback GTest::Main autodiff::autodiff)
  # gtest_discover_tests(test_asif)

  add_executable(test_ekf test/test_ekf.cpp)
  target_link_libraries(test_ekf PRIVATE feedback GTest::Main autodiff::autodiff)
  gtest_discover_tests(test_ekf)

  add_executable(test_mpc test/test_mpc.cpp)
  target_link_libraries(test_mpc PRIVATE feedback GTest::Main autodiff::autodiff)
  gtest_discover_tests(test_mpc)

  add_executable(test_ldlt test/test_ldlt.cpp)
  target_link_libraries(test_ldlt PRIVATE feedback lapacke GTest::Main)
  gtest_discover_tests(test_ldlt)

  add_executable(test_qp test/test_qp.cpp)
  target_link_libraries(test_qp PRIVATE feedback GTest::Main)
  gtest_discover_tests(test_qp)
endif()
