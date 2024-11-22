#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

cmake_minimum_required(VERSION 3.16)
include(FetchContent)

project(intel_npu_acceleration_library_example)

set(CMAKE_CXX_STANDARD 14)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-march=native)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


FetchContent_Declare(
        intel_npu_acceleration_library
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)
FetchContent_MakeAvailable(intel_npu_acceleration_library)


find_package(OpenVINO REQUIRED PATHS ${openvino_SOURCE_DIR}/runtime/cmake)

include_directories(${intel_npu_acceleration_library_SOURCE_DIR}/include)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(intel_npu_acceleration_library_example main.cpp)

target_link_libraries(intel_npu_acceleration_library_example PRIVATE openvino::runtime)
