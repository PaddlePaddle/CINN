cmake_minimum_required(VERSION 3.0)
project(CINN)

set(CMAKE_CXX_STANDARD 14)
set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/thirds")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WITH_TESTING         "Compile with Unittests"          ON)

# include the customized configures
include(${CMAKE_BINARY_DIR}/config.cmake)

include(cmake/core.cmake)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

#include(cmake/external/pybind11.cmake)
include(cmake/external/gflags.cmake)
include(cmake/external/glog.cmake)
include(cmake/external/gtest.cmake)
# include(cmake/external/protobuf.cmake)
# include(cmake/external/mklml.cmake)
#include(cmake/external/boost.cmakeg)
find_package(Threads REQUIRED)

include_directories(${ISL_HOME}/include)
find_library(isl_lib isl PATHS "${ISL_HOME}/lib" NO_DEFAULT_PATH)
link_directories(${ISL_HOME}/lib)

include_directories("/usr/include")
find_library(ginac_lib ginac PATHS "/usr/lib" NO_DEFAULT_PATH)
message(STATUS "ginac: ${ginac_lib}")
link_directories("/usr/lib")

set(core_src CACHE INTERNAL "" FORCE)
add_subdirectory(cinn)
add_subdirectory(tests)

cc_library(core SRCS ${core_src} DEPS glog)
