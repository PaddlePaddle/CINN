set(srcs
        kernel_frame.cc
        kernel_registry.cc
        value.cc
        kernel_utils.cc
  )

cc_test(test_host_context_value SRCS value_test.cc DEPS core)
cc_test(test_kernel_utils SRCS kernel_utils_test.cc DEPS core)

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/host_context/${cpp}"
    CACHE INTERNAL "")
endforeach()
