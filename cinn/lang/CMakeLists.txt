cc_library(lang SRCS
        buffer.cc
        tensor.cc
        compute.cc
        placeholder.cc
        DEPS ir poly runtime)

cc_test(test_compute SRCS compute_test.cc DEPS lang)
cc_test(test_tensor2 SRCS tensor_test.cc DEPS lang)
cc_test(test_placeholder SRCS placeholder_test.cc DEPS lang)
