find_package(gflags REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(osqp REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE ${GFLAGS_INCLUDE_DIR})
target_link_libraries(
  bench
  PRIVATE feedback
          gflags
          Matplot++::matplot
          osqp::osqpstatic
)
target_compile_options(
  bench
  PRIVATE -Wall
          -Wextra
          -Wpedantic
          -march=native
          -mtune=native
          -O3
          -DNDEBUG
)
