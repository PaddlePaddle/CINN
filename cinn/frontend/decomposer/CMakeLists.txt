core_gather_headers()

gather_srcs(cinnapi_src SRCS
    activation.cc
    )

cc_test(test_activation_decomposer SRCS activation_test.cc DEPS cinncore)
