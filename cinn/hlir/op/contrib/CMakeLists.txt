core_gather_headers()

gather_srcs(cinnapi_src SRCS
    pool_grad.cc
    )

cc_test(test_pool_grad SRCS pool_grad_test.cc DEPS cinncore)
