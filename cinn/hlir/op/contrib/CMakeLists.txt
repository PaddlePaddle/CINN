core_gather_headers()

gather_srcs(cinnapi_src SRCS
    clip.cc
    )

cc_test(test_clip SRCS clip_test.cc DEPS cinncore)
