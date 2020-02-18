cc_library(optim
        SRCS remove_nested_block.cc
        replace_call_with_expr.cc
        ir_copy.cc
        replace_var_with_expr.cc
        DEPS ir)

cc_test(test_remove_nested_block SRCS remove_nested_block_test.cc DEPS optim)
cc_test(test_ir_copy SRCS ir_copy_test.cc DEPS optim lang)
cc_test(test_replace_call_with_expr SRCS replace_call_with_expr_test.cc DEPS optim lang poly)
