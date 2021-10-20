core_gather_headers()

gather_srcs(cinnapi_src SRCS
    activation.cc
    elementwise.cc
    )

# disable the test on x86 due to "SubProcess abort"
if(WITH_CUDA)
  nv_test(test_activation_decomposer SRCS activation_test.cc DEPS cinncore)
endif()

cc_test(test_elementwise_decomposer SRCS elementwise_test.cc DEPS cinncore)
