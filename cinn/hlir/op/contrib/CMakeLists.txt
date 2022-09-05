core_gather_headers()

gather_srcs(cinnapi_src SRCS
        flip.cc
        cast.cc
        squeeze.cc
        clip.cc
        )

cc_test(test_cast SRCS cast_test.cc DEPS cinncore)
cc_test(test_squeeze SRCS squeeze_test.cc DEPS cinncore)
cc_test(test_clip SRCS clip_test.cc DEPS cinncore)
cc_test(test_flip SRCS flip_test.cc DEPS cinncore) 
