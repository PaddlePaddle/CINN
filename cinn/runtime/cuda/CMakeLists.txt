# install cuda/use_extern_funcs.h
set(core_includes "${core_includes};cinn/runtime/cuda/use_extern_funcs.h" CACHE INTERNAL "")

if (NOT WITH_CUDA)
    return()
endif()

set(srcs
  cuda_module.cc cuda_util.cc
  cuda_intrisics.cc
  )

set(includes
  cinn_cuda_runtime_source.cuh
  cuda_module.h
  cuda_util.h
  )

nv_test(test_cuda_module SRCS cuda_module_test.cc DEPS cinncore)
nv_library(cuda_runtime SRCS cinn_cuda_runtime_source.cuh)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinn/runtime/cuda/${cpp}"
            CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/runtime/cuda/${header}" CACHE INTERNAL "")
endforeach()
