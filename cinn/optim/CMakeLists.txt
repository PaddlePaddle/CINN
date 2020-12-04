set(srcs remove_nested_block.cc replace_call_with_expr.cc ir_copy.cc
  ir_replace.cc
  replace_var_with_expr.cc
  tensor_write_tell.cc
  ir_simplify.cc
  optimize.cc
  vectorize_loops.cc
  unroll_loops.cc
  transform_polyfor_to_for.cc
  eliminate_broadcast_in_forloop.cc
  fold_cinn_call_arguments.cc
  call_arg_list_to_pod_value.cc
  insert_debug_log_callee.cc
  lower_function_call_bind_vars.cc
  extern_call_process.cc
  map_extern_call.cc
  cache_read_write_replace.cc
  compute_inline_expand.cc
  buffer_assign.cc
  replace_const_param_to_integer.cc
  cast_simplify.cc
  compare_simplify.cc
  if_simplify.cc
  lower_intrin.cc
  cast_bool_to_int8.cc
  )

set(includes
    buffer_assign.h
    cache_read_write_replace.h
    call_arg_list_to_pod_value.h
    cast_simplify.h
    compare_simplify.h
    compute_inline_expand.h
    eliminate_broadcast_in_forloop.h
    extern_call_process.h
    fold_cinn_call_arguments.h
    if_simplify.h
    insert_debug_log_callee.h
    ir_copy.h
    ir_replace.h
    ir_simplify.h
    lower_function_call_bind_vars.h
    lower_intrin.h
    map_extern_call.h
    optimize.h
    remove_nested_block.h
    replace_call_with_expr.h
    replace_const_param_to_integer.h
    replace_var_with_expr.h
    tensor_write_tell.h
    transform_gpu_forloop.h
    transform_polyfor_to_for.h
    unroll_loops.h
    vectorize_loops.h
    )

if (WITH_CUDA)
  list(APPEND srcs transform_gpu_forloop.cc)
endif()

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/optim/${cpp}"
      CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/optim/${header}" CACHE INTERNAL "")
endforeach()

cc_test(test_remove_nested_block SRCS remove_nested_block_test.cc DEPS cinncore)
cc_test(test_ir_copy SRCS ir_copy_test.cc DEPS cinncore)
cc_test(test_ir_simplify SRCS ir_simplify_test.cc DEPS cinncore)
cc_test(test_replace_call_with_expr SRCS replace_call_with_expr_test.cc DEPS cinncore)
cc_test(test_vectorize_loops SRCS vectorize_loops_test.cc DEPS cinncore ARGS ${global_test_args})
cc_test(test_transform_polyfor_to_for SRCS transform_polyfor_to_for_test.cc DEPS cinncore ARGS ${global_test_args})
cc_test(test_optimize SRCS optimize_test.cc DEPS cinncore)
cc_test(test_cache_read_write_replace SRCS cache_read_write_replace_test.cc DEPS cinncore)
cc_test(test_cast_simplify SRCS cast_simplify_test.cc DEPS cinncore)
cc_test(test_compare_simplify SRCS compare_simplify_test.cc DEPS cinncore)
cc_test(test_if_simplify SRCS if_simplify_test.cc DEPS cinncore)

if (WITH_CUDA)
  cc_test(test_transform_gpu_forloop SRCS transform_gpu_forloop_test.cc DEPS cinncore)
endif()
