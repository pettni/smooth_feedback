find_package(osqp REQUIRED)
find_package(gflags REQUIRED)
find_package(Matplot++ REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-march=native -mtune=native -O3 -DNDEBUG)

add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE ${GFLAGS_INCLUDE_DIR})
target_link_libraries(bench PRIVATE feedback lapacke gflags Matplot++::matplot
                                    osqp::osqpstatic)
