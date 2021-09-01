find_package(autodiff REQUIRED)

add_executable(mpc_doubleintegrator mpc_doubleintegrator.cpp)
target_link_libraries(mpc_doubleintegrator autodiff::autodiff feedback)

add_executable(asif_doubleintegrator asif_doubleintegrator.cpp)
target_link_libraries(asif_doubleintegrator autodiff::autodiff feedback)

add_executable(mpc_se2 mpc_se2.cpp)
target_link_libraries(mpc_se2 autodiff::autodiff feedback)

add_executable(snippets snippets.cpp)
target_link_libraries(snippets feedback)

find_package(Matplot++ QUIET)
if(${Matplot++_FOUND})
  set(CMAKE_CXX_STANDARD 20)  # thanks Matplot++
  target_compile_definitions(mpc_doubleintegrator PRIVATE ENABLE_PLOTTING)
  target_link_libraries(mpc_doubleintegrator Matplot++::matplot)

  target_compile_definitions(asif_doubleintegrator PRIVATE ENABLE_PLOTTING)
  target_link_libraries(asif_doubleintegrator Matplot++::matplot)

  target_compile_definitions(mpc_se2 PRIVATE ENABLE_PLOTTING)
  target_link_libraries(mpc_se2 Matplot++::matplot)
else()
  message(WARNING "Matplot++ not found, plotting disabled")
endif()
