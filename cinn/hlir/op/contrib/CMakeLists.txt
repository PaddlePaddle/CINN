core_gather_headers()

gather_srcs(cinnapi_src SRCS
    clip.cc
    one_hot.cc
    )

cc_test(test_clip SRCS clip_test.cc DEPS cinncore)
cc_test(test_one_hot SRCS one_hot_test.cc DEPS cinncore)
