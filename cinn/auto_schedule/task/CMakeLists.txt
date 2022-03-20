core_gather_headers()

gather_srcs(cinnapi_src SRCS
    task_creator.cc
    tune_task.cc
    )

cc_test(test_task_creator SRCS task_creator_test.cc DEPS cinncore)
cc_test(test_tune_task SRCS tune_task_test.cc DEPS cinncore)
