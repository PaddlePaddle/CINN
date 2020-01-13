cmake_minimum_required(VERSION 3.0)
project(CINN)

set(CMAKE_CXX_STANDARD 11)
set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/thirds")

option(WITH_TESTING         "Compile with Unittests"          ON)

include(cmake/core.cmake)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

#include(cmake/external/pybind11.cmake)
include(cmake/external/gflags.cmake)
include(cmake/external/glog.cmake)
include(cmake/external/gtest.cmake)
include(cmake/external/protobuf.cmake)
include(cmake/external/mklml.cmake)

find_package(Threads REQUIRED)
add_subdirectory(cinn)
