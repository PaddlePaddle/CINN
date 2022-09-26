core_gather_headers()

gather_srcs(cinnapi_src SRCS
        gather.cc
        scatter.cc
        cast.cc
        squeeze.cc
        clip.cc
        arange.cc
        sort.cc
        squeeze.cc
        one_hot.cc
        )

cc_test(test_cast SRCS cast_test.cc DEPS cinncore)
cc_test(test_squeeze SRCS squeeze_test.cc DEPS cinncore)
cc_test(test_gather SRCS gather_test.cc DEPS cinncore)
cc_test(test_scatter SRCS scatter_test.cc DEPS cinncore)
cc_test(test_clip SRCS clip_test.cc DEPS cinncore)
cc_test(test_sort SRCS sort_test.cc DEPS cinncore)
cc_test(test_arange SRCS arange_test.cc DEPS cinncore)
cc_test(test_one_hot SRCS one_hot_test.cc DEPS cinncore)
