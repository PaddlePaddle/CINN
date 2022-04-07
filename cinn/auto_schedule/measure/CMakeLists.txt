core_gather_headers()

gather_srcs(cinnapi_src SRCS schedule_measurer.cc simple_builder.cc simple_runner.cc)

# TODO(CtfGo) this test not work in current and will be updated
# once GraphCompiler can return LoweredFunc directly
cc_test(test_simple_runner SRCS simple_runner_test.cc DEPS cinncore)
cc_test(test_measurer SRCS measurer_test.cc DEPS cinncore)
