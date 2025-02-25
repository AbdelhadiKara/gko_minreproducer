# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}")

project(gko_reproducer LANGUAGES C CXX)


set(CMAKE_C_COMPILER /opt/rocm-6.1.2/bin/amdclang)
set(CMAKE_CXX_COMPILER /opt/rocm-6.1.2/bin/amdclang++ )# hipcc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)
# NOTE: We are not supposed to define CMAKE_BUILD_TYPE here.
set(CMAKE_BUILD_TYPE Debug) # Debug, Release, RelWithDebInfo and MinSizeRel


find_package(Kokkos REQUIRED)
find_package(Ginkgo REQUIRED)
add_executable(gko_test debug_gko.cpp) 
target_link_libraries(gko_test PRIVATE Kokkos::kokkos Ginkgo::ginkgo)




