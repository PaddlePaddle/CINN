cc_library(common
        SRCS shared.cc pod_value.cc type.cc
        DEPS boost)

cc_test(test_pod_value
        SRCS pod_value_test.cc DEPS common)
