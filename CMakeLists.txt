cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(CINN)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/thirds")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(DOWNLOAD_MODEL_DIR "${THIRD_PARTY_PATH}/model")

option(WITH_TESTING         "Compile with Unittests"                OFF)
option(WITH_INFRT           "Whether to build infrt"                OFF)
option(WITH_MKL_CBLAS       "Compile MKL with CBLAS support"        ON)
option(WITH_MKLDNN          "Compile MKLDNN support"                ON)
option(WITH_CUDA            "Compile with CUDA support"             OFF)
option(WITH_CUDNN           "Compile with CUDNN support"            OFF)
option(WITH_DEBUG           "Compile with debug information"        OFF)
option(PUBLISH_LIBS         "Whether to publish compiled libraries" ON)
option(PY_VERSION           "Python version"                        ${PY_VERSION})

if (NOT PY_VERSION)
  set(PY_VERSION 3.6)
endif()

if (WITH_TESTING)
  add_definitions(-DCINN_WITH_TEST)
endif()
if (WITH_DEBUG)
  add_definitions(-DCINN_WITH_DEBUG)
endif()

include(cmake/version.cmake)
# include the customized configures
if (EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
endif()

if (WITH_CUDA)
  message(STATUS "Enable CUDA")
  add_definitions(-DCINN_WITH_CUDA)
  if (WITH_CUDNN)
    message(STATUS "Enable CUDNN")
    add_definitions(-DCINN_WITH_CUDNN)
  endif()
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(${CMAKE_SOURCE_DIR}/cinn/runtime/cuda)
  include_directories(/usr/lib/x86_64-linux-gnu)
  set(CUDA_SEPARABLE_COMPILATION ON)

  find_library(CUDASTUB libcuda.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/ REQUIRED)
  find_library(CUBLAS libcublas.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/lib REQUIRED)
  find_library(CUDNN libcudnn.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/lib REQUIRED)
endif()

find_package(Threads REQUIRED)

set(cinnapi_src CACHE INTERNAL "" FORCE)
set(infrt_src CACHE INTERNAL "" FORCE)
set(core_src CACHE INTERNAL "" FORCE)
set(core_includes CACHE INTERNAL "" FORCE)
set(core_proto_includes CACHE INTERNAL "" FORCE)
include(cmake/core.cmake)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

include(cmake/external/pybind11.cmake)
include(cmake/external/gflags.cmake)
include(cmake/external/glog.cmake)
include(cmake/external/gtest.cmake)
include(cmake/external/absl.cmake)
include(cmake/nvrtc.cmake)
include(cmake/nvtx.cmake)
include(cmake/external/llvm.cmake)
include(cmake/external/isl.cmake)
include(cmake/external/ginac.cmake)
include(cmake/external/protobuf.cmake)
include(cmake/external/mklml.cmake)
include(cmake/external/mkldnn.cmake)
include(cmake/external/openmp.cmake)
find_package(Threads REQUIRED)

set(LINK_FLAGS "-Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/cmake/export.map" CACHE INTERNAL "")
set(global_test_args "--cinn_x86_builtin_code_root=${CMAKE_SOURCE_DIR}/cinn/backends")

set(core_deps CACHE INTERNAL "" FORCE)
set(hlir_src CACHE INTERNAL "" FORCE)
add_subdirectory(cinn)
if (WITH_INFRT)
add_subdirectory(infrt)
endif()
add_subdirectory(tests)
add_subdirectory(tutorials)

set(core_src "${cinnapi_src}")

set(Python_VIRTUALENV FIRST)
find_package(PythonInterp ${PY_VERSION} REQUIRED)
find_package(PythonLibs ${PY_VERSION} REQUIRED)
message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIR})
cc_library(cinnapi SHARED SRCS ${cinnapi_src} DEPS glog ${llvm_libs} framework_proto param_proto framework_proto absl isl ginac pybind)
add_dependencies(cinnapi GEN_LLVM_RUNTIME_IR_HEADER ZLIB::ZLIB)
add_dependencies(cinnapi GEN_LLVM_RUNTIME_IR_HEADER ${core_deps})

target_link_libraries(cinnapi ${PYTHON_LIBRARIES})

if (WITH_MKL_CBLAS)
  target_link_libraries(cinnapi mklml)
  add_dependencies(cinnapi mklml)
  if (WITH_MKLDNN)
    target_link_libraries(cinnapi mkldnn)
    add_dependencies(cinnapi mkldnn)
  endif()
endif()

if (WITH_CUDA)
  target_link_libraries(cinnapi ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES} ${CUDASTUB} ${CUBLAS} ${CUDNN})
  if (NVTX_FOUND)
    target_link_libraries(cinnapi ${CUDA_NVTX_LIB})
  endif()
endif()

function(gen_cinncore LINKTYPE)
  set(CINNCORE_TARGET cinncore)
  if (${LINKTYPE} STREQUAL "STATIC")
    set(CINNCORE_TARGET cinncore_static)
  endif()
  cc_library(${CINNCORE_TARGET} ${LINKTYPE} SRCS ${core_src} DEPS glog ${llvm_libs} framework_proto param_proto framework_proto absl isl ginac)
  add_dependencies(${CINNCORE_TARGET} GEN_LLVM_RUNTIME_IR_HEADER ZLIB::ZLIB)
  add_dependencies(${CINNCORE_TARGET} GEN_LLVM_RUNTIME_IR_HEADER ${core_deps})

  add_dependencies(${CINNCORE_TARGET} pybind)
  target_link_libraries(${CINNCORE_TARGET} ${PYTHON_LIBRARIES})

  if (WITH_MKL_CBLAS)
    target_link_libraries(${CINNCORE_TARGET} mklml)
    add_dependencies(${CINNCORE_TARGET} mklml)
    if (WITH_MKLDNN)
      target_link_libraries(${CINNCORE_TARGET} mkldnn)
      add_dependencies(${CINNCORE_TARGET} mkldnn)
    endif()
  endif()

  if (WITH_CUDA)
    target_link_libraries(${CINNCORE_TARGET} ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES} ${CUDASTUB} ${CUBLAS} ${CUDNN})
    if (NVTX_FOUND)
      target_link_libraries(${CINNCORE_TARGET} ${CUDA_NVTX_LIB})
    endif()
  endif()
endfunction()

gen_cinncore(STATIC)
gen_cinncore(SHARED)

# MLIR td file generations
if (WITH_INFRT)
set(infrt_mlir_incs
        ops_inc
        basic_kernels_inc
        test_kernels_inc
        cinn_base_inc
        tensor_shape_inc
        dense_tensor_inc
        pd_ops_inc
        rewrite_inc
        )

cc_library(infrt SRCS ${infrt_src} DEPS glog absl paddle_framework_proto ${mlir_libs})
add_dependencies(infrt ${infrt_mlir_incs})
endif ()

# --------distribute cinncore lib and include begin--------
if (PUBLISH_LIBS)
    set(core_includes "${core_includes};cinn/runtime/cuda/cinn_cuda_runtime_source.cuh")
    foreach(header ${core_includes})
        get_filename_component(prefix ${header} DIRECTORY)
        file(COPY ${header} DESTINATION ${CMAKE_BINARY_DIR}/dist/cinn/include/${prefix})
    endforeach()

    foreach(proto_header ${core_proto_includes})
      string(REPLACE ${CMAKE_BINARY_DIR}/ "" relname ${proto_header})
      get_filename_component(prefix ${relname} DIRECTORY)
      set(target_name ${CMAKE_BINARY_DIR}/dist/cinn/include/${relname})
      add_custom_command(TARGET cinnapi POST_BUILD
        COMMENT "copy generated proto header '${relname}' to dist"
        COMMAND cmake -E copy ${proto_header} ${target_name} DEPENDS cinnapi)
    endforeach()

    add_custom_command(TARGET cinnapi POST_BUILD
        COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/libcinnapi.so ${CMAKE_BINARY_DIR}/dist/cinn/lib/libcinnapi.so
        COMMAND cmake -E copy_directory ${CMAKE_BINARY_DIR}/thirds/install ${CMAKE_BINARY_DIR}/dist/third_party
        DEPENDS cinnapi
    )
    add_custom_command(TARGET cinncore_static POST_BUILD
        COMMAND cmake -E copy ${CMAKE_SOURCE_DIR}/tutorials/demo.cc ${CMAKE_BINARY_DIR}/dist/demo.cc
        COMMAND cmake -E copy ${CMAKE_SOURCE_DIR}/tutorials/build_demo.sh ${CMAKE_BINARY_DIR}/dist/build_demo.sh
        COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/libcinncore_static.a ${CMAKE_BINARY_DIR}/dist/cinn/lib/libcinncore_static.a
        COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/cinn/frontend/paddle/libframework_proto.a ${CMAKE_BINARY_DIR}/dist/cinn/lib/libframework_proto.a
        COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/cinn/hlir/pe/libparam_proto.a ${CMAKE_BINARY_DIR}/dist/cinn/lib/libparam_proto.a
        COMMENT "distribute libcinncore_static.a and related header files."
        DEPENDS cinncore_static
    )
endif()
# --------distribute cinncore lib and include end--------

add_subdirectory(python)
