core_gather_headers()

core_gather_srcs(SRCS
    nn.cc
    broadcast.cc
    transform.cc
    elementwise.cc
    reduction.cc
    op_util.cc
    )

cc_test(test_cinn_op_broadcast SRCS op_broadcast_test.cc DEPS cinncore)
cc_test(test_cinn_op_nn SRCS op_nn_test.cc DEPS cinncore)
