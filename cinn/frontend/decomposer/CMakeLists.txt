core_gather_headers()

gather_srcs(cinnapi_src SRCS
    activation.cc
    nn.cc
    )

cc_test(test_decomposer
        SRCS decomposer_test.cc DEPS cinncore)
