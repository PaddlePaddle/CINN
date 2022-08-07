core_gather_headers()

gather_srcs(cinnapi_src SRCS
    argmin.cc
    argmax.cc
    )

cc_test(test_argmin SRCS argmin_test.cc DEPS cinncore)
cc_test(test_argmax SRCS argmax_test.cc DEPS cinncore)