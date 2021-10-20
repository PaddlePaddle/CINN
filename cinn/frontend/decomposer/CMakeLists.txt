core_gather_headers()

gather_srcs(cinnapi_src SRCS
    activation.cc
    nn.cc
    )

if(WITH_CUDA)
  nv_test(test_activation_decomposer SRCS activation_test.cc DEPS cinncore)
endif()

cc_test(test_nn_decomposer
        SRCS nn_test.cc DEPS cinncore)
