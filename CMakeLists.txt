cmake_minimum_required(VERSION 3.0)
project(CINN)

set(CMAKE_CXX_STANDARD 17)

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/thirds")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WITH_TESTING         "Compile with Unittests"          ON)
option(WITH_TESTS           "Compile with tests"              ON)

# include the customized configures
include(${CMAKE_BINARY_DIR}/config.cmake)

if (WITH_CUDA)
  message(STATUS "Enable CUDA")
  add_definitions(-DCINN_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CUDA_SEPARABLE_COMPILATION ON)
endif()

if (WITH_MKL_CBLAS)
    add_definitions(-DCINN_WITH_MKL_CBLAS)
endif()

find_package(Threads REQUIRED)


include(cmake/core.cmake)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

include(cmake/external/pybind11.cmake)
include(cmake/external/gflags.cmake)
include(cmake/external/glog.cmake)
include(cmake/external/gtest.cmake)
include(cmake/nvrtc.cmake)
include(cmake/llvm.cmake)
# include(cmake/external/protobuf.cmake)
include(cmake/external/mklml.cmake)
#include(cmake/external/boost.cmakeg)
find_package(Threads REQUIRED)

include_directories(${ISL_HOME}/include)
find_library(isl_lib isl PATHS "${ISL_HOME}/lib" NO_DEFAULT_PATH)
link_directories(${ISL_HOME}/lib)

include_directories("/usr/include")
find_library(ginac_lib ginac PATHS "/usr/lib" NO_DEFAULT_PATH)
message(STATUS "ginac: ${ginac_lib}")
link_directories("/usr/lib")

set(global_test_args "--cinn_x86_builtin_code_root=${CMAKE_SOURCE_DIR}/cinn/backends")

set(core_src CACHE INTERNAL "" FORCE)
set(hlir_src CACHE INTERNAL "" FORCE)
add_subdirectory(cinn)
add_subdirectory(tests)

cc_library(core SRCS ${core_src} DEPS glog LLVM mklml)
add_dependencies(core GEN_LLVM_RUNTIME_IR_HEADER)

if (WITH_CUDA)
  target_link_libraries(core ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES})
endif()

add_subdirectory(python)
