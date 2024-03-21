# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.26)

project(gko_reproducer LANGUAGES CXX)
find_package(Ginkgo 1.7.0  REQUIRED)

add_executable(gko_test
  	debug_gko.cpp
)
target_link_libraries(gko_test PUBLIC Ginkgo::ginkgo)