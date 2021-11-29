core_gather_headers()

gather_srcs(cinnapi_src SRCS
    decomposer.cc
    remove_identity.cc
    )


cc_test(test_decomposer_pass SRCS decomposer_test.cc DEPS cinncore)
