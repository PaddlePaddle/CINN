core_gather_headers()
gather_srcs(cinnapi_src SRCS
    sigmoid.cc
    fetch_feed.cc
    scale.cc
    mul.cc
    relu.cc
    softmax.cc
    elementwise.cc
    conv2d.cc
    pool2d.cc
    batchnorm.cc
    slice.cc
    dropout.cc
    transpose.cc
    reshape.cc
    tanh.cc
    matmul.cc
    compare.cc
    log.cc
    concat.cc
    constant.cc
    where.cc
    squeeze.cc
    clip.cc
    layer_norm.cc
    expand.cc)

cc_test(test_paddle_layer_norm SRCS layer_norm_test.cc DEPS cinncore)
cc_test(test_paddle_expand SRCS expand_test.cc DEPS cinncore)
