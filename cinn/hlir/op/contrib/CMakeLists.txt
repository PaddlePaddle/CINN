core_gather_headers()

gather_srcs(cinnapi_src SRCS
        sort.cc
    )

cc_test(test_sort SRCS sort_test.cc DEPS cinncore)
