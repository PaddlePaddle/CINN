cc_library(lang SRCS
        buffer.cc
        compute.cc
        placeholder.cc
        tensor.cc
        module.cc
        DEPS ir poly runtime)

cc_test(test_compute SRCS compute_test.cc DEPS lang)
cc_test(test_placeholder SRCS placeholder_test.cc DEPS lang)
cc_test(test_tensor SRCS tensor_test.cc DEPS lang)
