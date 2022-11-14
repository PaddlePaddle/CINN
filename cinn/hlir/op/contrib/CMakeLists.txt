core_gather_headers()

gather_srcs(cinnapi_src SRCS
        scatter.cc
        arange.cc
        flip.cc
        sort.cc
        argmin.cc
        argmax.cc
        repeat.cc
        lookup_table.cc
        one_hot.cc
        )

cc_test(test_scatter SRCS scatter_test.cc DEPS cinncore)
cc_test(test_sort SRCS sort_test.cc DEPS cinncore)
cc_test(test_argmin SRCS argmin_test.cc DEPS cinncore)
cc_test(test_argmax SRCS argmax_test.cc DEPS cinncore)
cc_test(test_arange SRCS arange_test.cc DEPS cinncore)
cc_test(test_flip SRCS flip_test.cc DEPS cinncore)
cc_test(test_repeat SRCS repeat_test.cc DEPS cinncore)
cc_test(test_one_hot SRCS one_hot_test.cc DEPS cinncore)
cc_test(test_lookup_table SRCS lookup_table_test.cc DEPS cinncore)
