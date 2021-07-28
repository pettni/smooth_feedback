find_package(Matplot++ REQUIRED)

set(CMAKE_CXX_STANDARD 20)  # thanks Matplot++

add_executable(mpc_doubleintegrator mpc_doubleintegrator.cpp)
target_link_libraries(mpc_doubleintegrator lapacke feedback Matplot++::matplot)

add_executable(snippets snippets.cpp)
target_link_libraries(snippets feedback)
