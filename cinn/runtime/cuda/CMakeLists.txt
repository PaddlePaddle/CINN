if (NOT WITH_CUDA)
    return()
endif()

core_gather_headers()

core_gather_srcs(SRCS
  cuda_module.cc
  cuda_util.cc
  cuda_intrinsics.cc
  )

gather_srcs(cinnapi_src SRCS
  cuda_module.cc
  cuda_util.cc
  cuda_intrinsics.cc
  )

nv_test(test_cuda_module SRCS cuda_module_test.cc DEPS cinncore)
nv_library(cuda_runtime SRCS cinn_cuda_runtime_source.cuh)
