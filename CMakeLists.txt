cmake_minimum_required(VERSION 3.0)
project(CINN)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/thirds")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(DOWNLOAD_MODEL_DIR "${THIRD_PARTY_PATH}/model")

option(WITH_TESTING         "Compile with Unittests"          ON)
option(WITH_CUDA            "Compile with CUDA support"       OFF)
option(WITH_DEBUG           "Compile with debug information"  OFF)
option(PUBLISH_LIBS         "Whether to publish compiled libraries" OFF)

if (WITH_TESTING)
  add_definitions(-DCINN_WITH_TEST)
endif()
if (WITH_DEBUG)
  add_definitions(-DCINN_WITH_DEBUG)
endif()

# include the customized configures
include(${CMAKE_BINARY_DIR}/config.cmake)

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
endif()

if (WITH_MKL_CBLAS)
    add_definitions(-DCINN_WITH_MKL_CBLAS)
endif()

find_package(Threads REQUIRED)


set(cinnapi_src CACHE INTERNAL "" FORCE)
set(cinnrt_src CACHE INTERNAL "" FORCE)
set(core_src CACHE INTERNAL "" FORCE)
set(core_includes CACHE INTERNAL "" FORCE)
include(cmake/core.cmake)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

include(cmake/external/pybind11.cmake)
include(cmake/external/gflags.cmake)
include(cmake/external/glog.cmake)
include(cmake/external/gtest.cmake)
include(cmake/external/absl.cmake)
include(cmake/nvrtc.cmake)
include(cmake/llvm.cmake)
include(cmake/external/protobuf.cmake)
include(cmake/external/mklml.cmake)
include(cmake/external/mkldnn.cmake)
include(cmake/external/openmp.cmake)
find_package(Threads REQUIRED)

include_directories(${ISL_HOME}/include)
add_library(isl STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET isl PROPERTY IMPORTED_LOCATION "${ISL_HOME}/lib/libisl.a")

include_directories("/usr/local/include")
add_library(gmp STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gmp PROPERTY IMPORTED_LOCATION /usr/local/lib/libgmp.a)

add_library(cln STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET cln PROPERTY IMPORTED_LOCATION /usr/local/lib/libcln.a)

add_library(ginac STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET ginac PROPERTY IMPORTED_LOCATION /usr/local/lib/libginac.a)

set(LINK_FLAGS "-Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/cmake/export.map" CACHE INTERNAL "")
set(global_test_args "--cinn_x86_builtin_code_root=${CMAKE_SOURCE_DIR}/cinn/backends")

set(core_deps CACHE INTERNAL "" FORCE)
set(hlir_src CACHE INTERNAL "" FORCE)
add_subdirectory(cinn)
add_subdirectory(cinnrt)
add_subdirectory(tests)
add_subdirectory(tutorials)

set(core_src "${cinnapi_src};${cinnrt_src}")

set(cinnrt_mlir_incs
        ops_inc
        basic_kernels_inc
        test_kernels_inc
        cinn_base_inc
        tensor_shape_inc
        dense_tensor_inc
        pd_ops_inc
        rewrite_inc
        )


cc_library(cinnapi SHARED SRCS ${cinnapi_src} DEPS glog ${llvm_libs} mklml mkldnn framework_proto param_proto paddle_framework_proto ${mlir_libs} ${ABSL_LIBS} isl ginac cln gmp)
add_dependencies(cinnapi GEN_LLVM_RUNTIME_IR_HEADER ZLIB::ZLIB)
add_dependencies(cinnapi GEN_LLVM_RUNTIME_IR_HEADER
        # MLIR td file generations
        ${cinnrt_mlir_incs}
        ${core_deps}
        )
if (WITH_CUDA)
  target_link_libraries(cinnapi ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcudnn.so  /usr/lib/x86_64-linux-gnu/libcublas.so)
endif()

cc_library(cinncore SRCS ${core_src} DEPS glog ${llvm_libs} mklml mkldnn framework_proto param_proto paddle_framework_proto ${mlir_libs} ${ABSL_LIBS} isl ginac cln gmp)

cc_library(cinnrt SRCS ${cinnrt_src} DEPS glog ${ABSL_LIBS} ${mlir_libs})
add_dependencies(cinnrt ${cinnrt_mlir_incs})

if (TARGET LLVM)
  target_link_libraries(cinncore LLVM)
else()
  target_link_libraries(cinncore ${llvm_libs})
endif()
add_dependencies(cinncore GEN_LLVM_RUNTIME_IR_HEADER ZLIB::ZLIB)
add_dependencies(cinncore GEN_LLVM_RUNTIME_IR_HEADER
        # MLIR td file generations
        ops_inc
        basic_kernels_inc
        test_kernels_inc
        cinn_base_inc
        tensor_shape_inc
        dense_tensor_inc
        pd_ops_inc
        ${core_deps}
        )

if (WITH_CUDA)
  target_link_libraries(cinncore ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcudnn.so  /usr/lib/x86_64-linux-gnu/libcublas.so)
endif()

# --------distribute cinncore lib and include begin--------
if (PUBLISH_LIBS)
    foreach(header ${core_includes})
        get_filename_component(prefix ${header} DIRECTORY)
        file(COPY ${header} DESTINATION ${CMAKE_BINARY_DIR}/dist/cinn/include/${prefix})
    endforeach()

    add_custom_command(TARGET cinnapi POST_BUILD
        COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/libcinnapi.so ${CMAKE_BINARY_DIR}/dist/cinn/lib/libcinnapi.so
        DEPENDS cinnapi
    )
    add_custom_command(TARGET cinncore POST_BUILD
        COMMAND cmake -E copy ${CMAKE_SOURCE_DIR}/tutorials/demo.cc ${CMAKE_BINARY_DIR}/dist/demo.cc
        COMMAND cmake -E copy ${CMAKE_SOURCE_DIR}/tutorials/build_demo.sh ${CMAKE_BINARY_DIR}/dist/build_demo.sh
        COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/libcinncore.a ${CMAKE_BINARY_DIR}/dist/cinn/lib/libcinncore.a
        COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/cinn/frontend/paddle/libframework_proto.a ${CMAKE_BINARY_DIR}/dist/cinn/lib/libframework_proto.a
        COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/cinn/hlir/pe/libparam_proto.a ${CMAKE_BINARY_DIR}/dist/cinn/lib/libparam_proto.a
        COMMAND cmake -E copy_directory ${LLVM11_DIR} ${CMAKE_BINARY_DIR}/dist/third_party/llvm11
        COMMAND cmake -E copy_directory ${CMAKE_BINARY_DIR}/thirds/install ${CMAKE_BINARY_DIR}/dist/third_party
        COMMENT "distribute libcinncore.a and related header files."
        DEPENDS cinncore
    )
endif()
# --------distribute cinncore lib and include end--------

add_subdirectory(python)
