core_gather_headers()

gather_srcs(cinnapi_src SRCS task_creator.cc task_optimizer.cc)

cc_test(test_task_creator SRCS task_creator_test.cc DEPS cinncore)
