core_gather_headers()

gather_srcs(cinnapi_src SRCS
    activation.cc
    elementwise.cc
    broadcast.cc
    batch_norm.cc
    conv2d_grad.cc
    test_helper.cc
    )

cc_test(test_activation_decomposer SRCS activation_test.cc DEPS cinncore)
cc_test(test_elementwise_decomposer SRCS elementwise_test.cc DEPS cinncore)
cc_test(test_broadcast_decomposer SRCS broadcast_test.cc DEPS cinncore)
cc_test(test_batch_norm_decomposer SRCS batch_norm_test.cc DEPS cinncore)
if(WITH_CUDNN)
  cc_test(test_conv2d_grad_decomposer SRCS conv2d_grad_test.cc DEPS cinncore)
endif()
