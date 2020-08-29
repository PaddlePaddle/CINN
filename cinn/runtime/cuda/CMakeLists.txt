if (NOT WITH_CUDA)
    return()
endif()

set(srcs
  cuda_module.cc cuda_util.cc
  cuda_intrisics.cc
  )

nv_test(test_cuda_module SRCS cuda_module_test.cc DEPS core)
nv_library(cuda_runtime SRCS cinn_cuda_runtime_source.cuh)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinn/runtime/cuda/${cpp}"
            CACHE INTERNAL "")
endforeach()
