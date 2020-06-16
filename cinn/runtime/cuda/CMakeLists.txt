if (NOT WITH_CUDA)
    return()
endif()

set(srcs
  cuda_module.cc cuda_util.cc
  )

nv_test(test_cuda_module SRCS cuda_module_test.cc DEPS core)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinn/runtime/cuda/${cpp}"
            CACHE INTERNAL "")
endforeach()
