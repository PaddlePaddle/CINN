core_gather_headers()

gather_srcs(cinnapi_src SRCS
    flip.cc
    )

cc_test(test_flip SRCS flip_test.cc DEPS cinncore) 
