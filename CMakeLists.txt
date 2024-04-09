# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.26)

project(gko_reproducer LANGUAGES CXX)
# find_package(Kokkos REQUIRED)
add_subdirectory(tpls/kokkos)
find_package(Ginkgo 1.7.0  REQUIRED)
add_executable(gko_collmat
  	collmatrix_converge.cpp

)
target_link_libraries(gko_collmat PUBLIC Ginkgo::ginkgo Kokkos::kokkos)




