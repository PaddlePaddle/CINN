core_gather_headers()

gather_srcs(cinnapi_src SRCS
    squeeze.cc
    )

cc_test(test_squeeze SRCS squeeze_test.cc DEPS cinncore)
