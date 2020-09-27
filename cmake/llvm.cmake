if (${CMAKE_CXX_COMPILER} STREQUAL "clang++")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

message(STATUS "Found MLIR: ${MLIR_DIR}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# To build with MLIR, the LLVM is build from source code using the following flags:
set(MLIR_CMAKE_COMMAND
  " \
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_RTTI=ON \
")

# the valid tag is llvmorg-11.0.0-rc1
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs Support Core irreader
        X86 executionengine orcjit mcjit NVPTX AMDGPU all codegen)

message(STATUS "LLVM libs: ${llvm_libs}")

get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
message(STATUS "MLIR libs: ${mlir_libs}")
