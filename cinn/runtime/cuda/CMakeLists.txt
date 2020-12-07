file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()

if (NOT WITH_CUDA)
    return()
endif()

set(srcs
  cuda_module.cc cuda_util.cc
  cuda_intrisics.cc
  )

nv_test(test_cuda_module SRCS cuda_module_test.cc DEPS cinncore)
nv_library(cuda_runtime SRCS cinn_cuda_runtime_source.cuh)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinn/runtime/cuda/${cpp}"
            CACHE INTERNAL "")
endforeach()
