core_gather_headers()

gather_srcs(cinnapi_src SRCS
    activation.cc
    )

# disable the test on x86 due to "SubProcess abort"
if(WITH_CUDA)
  nv_test(test_activation_decomposer SRCS activation_test.cc DEPS cinncore)
endif()
