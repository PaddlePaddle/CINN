core_gather_headers()

gather_srcs(cinnapi_src SRCS
  flags.cc
  intrinsic.cc
  cinn_runtime.cc
  intrinsic_types.cc
  )

cc_library(cinn_runtime SRCS cinn_runtime.cc buffer.cc
        #cinn_x86_device_impl.cc
        )

cc_library(tiny_runtime STATIC SRCS tiny_runtime.cc)
cc_test(test_cinn_runtime SRCS cinn_runtime_test.cc DEPS cinn_runtime)

add_subdirectory(cuda)
add_subdirectory(cpu)
