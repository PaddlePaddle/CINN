core_gather_headers()

gather_srcs(cinnapi_src SRCS
        gather.cc
        scatter.cc
        cast.cc
        squeeze.cc
        clip.cc
        arange.cc
        flip.cc
        sort.cc
        argmin.cc
        argmax.cc
        squeeze.cc
        )

cc_test(test_cast SRCS cast_test.cc DEPS cinncore)
cc_test(test_squeeze SRCS squeeze_test.cc DEPS cinncore)
cc_test(test_gather SRCS gather_test.cc DEPS cinncore)
cc_test(test_scatter SRCS scatter_test.cc DEPS cinncore)
cc_test(test_clip SRCS clip_test.cc DEPS cinncore)
cc_test(test_sort SRCS sort_test.cc DEPS cinncore)
cc_test(test_argmin SRCS argmin_test.cc DEPS cinncore)
cc_test(test_argmax SRCS argmax_test.cc DEPS cinncore)
cc_test(test_arange SRCS arange_test.cc DEPS cinncore)
cc_test(test_flip SRCS flip_test.cc DEPS cinncore)
