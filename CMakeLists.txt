# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}")

project(gko_reproducer LANGUAGES C CXX)


#set(CMAKE_C_COMPILER /opt/rocm-6.2.1/bin/amdclang)
#set(CMAKE_CXX_COMPILER /opt/rocm-6.2.1/bin/hipcc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)
# NOTE: We are not supposed to define CMAKE_BUILD_TYPE here.
set(CMAKE_BUILD_TYPE Debug) # Debug, Release, RelWithDebInfo and MinSizeRel


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kokkos/cmake")
add_subdirectory("kokkos")
# Kokkos options:
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "")
set(Kokkos_ARCH_ZEN3 OFF CACHE BOOL "Optimize for AMD Zen3 architecture (HOST).")
set(Kokkos_ENABLE_HIP ON CACHE BOOL "Defined if the HIP execution space is enabled.")
set(Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS OFF CACHE BOOL "If defined, multiple kernel versions are instantiated potentially improving run time.")
set(Kokkos_ARCH_VEGA90A ON CACHE BOOL "Enable support for AMD GPU MI200 series (GFX90A).")
set(Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE OFF CACHE BOOL "")

add_executable(gko_test debug_gko.cpp)

include_directories("/lus/work/CT5/gen2224/akara/ginkgo-1.9.0/include" "/lus/work/CT5/gen2224/akara/build-ginkgo/include" "/lus/work/CT5/gen2224/akara/gko_minreproducer/kokkos/core/src" "/lus/work/CT5/gen2224/akara/gko_minreproducer/kokkos/tpls/desul/include" "/lus/work/CT5/gen2224/akara/gko_minreproducer/kokkos/tpls/mdspan/include"
"/lus/work/CT5/gen2224/akara/gko_minreproducer/gpu_build/kokkos/core/src/" "/lus/work/CT5/gen2224/akara/gko_minreproducer/gpu_build/kokkos")


target_link_libraries(gko_test INTERFACE Ginkgo::ginkgo Kokkos::kokkos)




