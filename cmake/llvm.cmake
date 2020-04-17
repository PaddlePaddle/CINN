if (${CMAKE_CXX_COMPILER} STREQUAL "clang++")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})

