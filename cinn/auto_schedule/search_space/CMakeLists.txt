add_subdirectory(auto_gen_rule)

core_gather_headers()

gather_srcs(cinnapi_src SRCS
    search_space.cc
    search_state.cc
    )

cc_test(test_search_space SRCS search_space_test.cc DEPS cinncore)
