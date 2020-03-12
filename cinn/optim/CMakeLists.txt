set(srcs remove_nested_block.cc replace_call_with_expr.cc ir_copy.cc
  replace_var_with_expr.cc
  tensor_write_tell.cc
  ir_simplify.cc
  optimize.cc
  )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/optim/${cpp}"
      CACHE INTERNAL "")
endforeach()

cc_test(test_remove_nested_block SRCS remove_nested_block_test.cc DEPS core)
cc_test(test_ir_copy SRCS ir_copy_test.cc DEPS core)
cc_test(test_ir_simplify SRCS ir_simplify_test.cc DEPS core)
cc_test(test_replace_call_with_expr SRCS replace_call_with_expr_test.cc DEPS
        core)
