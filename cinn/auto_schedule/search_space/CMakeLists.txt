add_subdirectory(auto_gen_rule)

core_gather_headers()

gather_srcs(cinnapi_src SRCS
    search_space.cc
    search_state.cc
    block_scheduler.cc
    rule_scheduler.cc
    )

cc_test(test_search_space SRCS search_space_test.cc DEPS cinncore)
cc_test(test_block_scheduler SRCS block_scheduler_test.cc DEPS cinncore)
cc_test(test_rule_scheduler SRCS rule_scheduler_test.cc DEPS cinncore)
