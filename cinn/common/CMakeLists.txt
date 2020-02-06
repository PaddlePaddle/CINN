cc_library(common
        SRCS shared.cc pod_value.cc type.cc
        object.cc
        graph_utils.cc
        DEPS boost utils)

cc_test(test_pod_value SRCS pod_value_test.cc DEPS common ir)
cc_test(test_shared SRCS shared_test.cc DEPS common)
cc_test(test_graph_utils SRCS graph_utils_test.cc DEPS common)
