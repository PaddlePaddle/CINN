add_subdirectory(cache)
add_subdirectory(cost_model)
add_subdirectory(measure)
add_subdirectory(search_space)
add_subdirectory(search_strategy)
add_subdirectory(task)
add_subdirectory(task_scheduler)

core_gather_headers()

gather_srcs(cinnapi_src SRCS auto_tuner.cc)

cc_test(test_auto_tuner SRCS auto_tuner_test.cc DEPS cinncore)
