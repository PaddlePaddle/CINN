core_gather_headers()

core_gather_srcs(SRCS
    instruction.cc
    )

cc_test(test_cinn_instruction SRCS instruction_test.cc DEPS cinncore)
