core_gather_headers()

gather_srcs(cinnapi_src SRCS cost_model.cc)

cc_test(test_cost_model SRCS cost_model_test.cc DEPS cinncore)
