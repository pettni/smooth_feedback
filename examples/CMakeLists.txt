find_package(autodiff REQUIRED)
find_package(Matplot++ QUIET)

set(CMAKE_CXX_STANDARD 20)  # thanks Matplot++

add_executable(mpc_doubleintegrator mpc_doubleintegrator.cpp)
target_link_libraries(mpc_doubleintegrator autodiff::autodiff lapacke feedback)
if(${Matplot++_FOUND})
  target_compile_definitions(mpc_doubleintegrator PRIVATE ENABLE_PLOTTING)
  target_link_libraries(mpc_doubleintegrator Matplot++::matplot)
else()
  message(WARN "Matplot++ not found, plotting disabled")
endif()

add_executable(snippets snippets.cpp)
target_link_libraries(snippets feedback)
