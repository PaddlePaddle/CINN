core_gather_headers()

gather_srcs(cinnapi_src SRCS
        cast.cc
        squeeze.cc
        clip.cc
        arange.cc
        sort.cc
        squeeze.cc
        argmin.cc
        argmax.cc
        )

cc_test(test_cast SRCS cast_test.cc DEPS cinncore)
cc_test(test_squeeze SRCS squeeze_test.cc DEPS cinncore)
cc_test(test_clip SRCS clip_test.cc DEPS cinncore)
cc_test(test_sort SRCS sort_test.cc DEPS cinncore)
cc_test(test_arange SRCS arange_test.cc DEPS cinncore)
cc_test(test_argmin SRCS argmin_test.cc DEPS cinncore)
cc_test(test_argmax SRCS argmax_test.cc DEPS cinncore)