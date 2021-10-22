core_gather_headers()

gather_srcs(cinnapi_src SRCS
    activation.cc
    elementwise.cc
    broadcast.cc
    batch_norm_train.cc
    )

cc_test(test_activation_decomposer SRCS activation_test.cc DEPS cinncore)
cc_test(test_elementwise_decomposer SRCS elementwise_test.cc DEPS cinncore)
cc_test(test_broadcast_decomposer SRCS broadcast_test.cc DEPS cinncore)

cc_test(test_batch_norm_train SRCS batch_norm_train_test.cc DEPS cinncore)
