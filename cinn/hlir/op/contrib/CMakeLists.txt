core_gather_headers()

gather_srcs(cinnapi_src SRCS
        cast.cc
        clip.cc
        sort.cc
    )

cc_test(test_cast SRCS cast_test.cc DEPS cinncore)
cc_test(test_clip SRCS clip_test.cc DEPS cinncore)
cc_test(test_sort SRCS sort_test.cc DEPS cinncore)
