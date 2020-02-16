cc_library(optim SRCS remove_nested_block.cc DEPS ir)

cc_test(test_remove_nested_block SRCS remove_nested_block_test.cc DEPS optim)
