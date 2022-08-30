core_gather_headers()

gather_srcs(cinnapi_src SRCS
        gather.cc
        scatter.cc
        cast.cc
        clip.cc
        )

cc_test(test_cast SRCS cast_test.cc DEPS cinncore)
cc_test(test_gather SRCS gather_test.cc DEPS cinncore)
cc_test(test_scatter SRCS scatter_test.cc DEPS cinncore)
cc_test(test_clip SRCS clip_test.cc DEPS cinncore)
