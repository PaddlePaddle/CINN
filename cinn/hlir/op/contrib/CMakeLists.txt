core_gather_headers()

gather_srcs(cinnapi_src SRCS
        cast.cc
    )

cc_test(test_cast SRCS cast_test.cc DEPS cinncore)
