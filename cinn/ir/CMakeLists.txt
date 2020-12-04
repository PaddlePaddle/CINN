set(srcs
    ir.cc
    ir_base.cc
    ir_visitor.cc
    ir_printer.cc
    ir_mutator.cc
    function_definition.cc
    lowered_func.cc
    ir_operators.cc
    ir_verify.cc
    buffer.cc
    function_base.cc
    operation.cc
    collect_ir_nodes.cc
    registry.cc
    tensor.cc
    module.cc
    intrinsic_ops.cc
    )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/ir/${cpp}"
      CACHE INTERNAL "")
endforeach()

set(includes
    buffer.h
    collect_ir_nodes.h
    function_base.h
    function_definition.h
    intrinsic_ops.h
    ir.h
    ir_base.h
    ir_mutator.h
    ir_operators.h
    ir_printer.h
    ir_verify.h
    ir_visitor.h
    lowered_func.h
    module.h
    operation.h
    registry.h
    tensor.h
    )

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/ir/${header}" CACHE INTERNAL "")
endforeach()

# cc_test(test_ir SRCS ir_test.cc DEPS core)
# cc_test(test_ir_printer SRCS ir_printer_test.cc DEPS core)
# cc_test(test_ir_operators SRCS ir_operators_test.cc DEPS core)
cc_test(test_collect_ir_nodes SRCS collect_ir_nodes_test.cc DEPS cinncore)
# cc_test(test_tensor SRCS tensor_test.cc DEPS core)
cc_test(test_buffer SRCS buffer_test.cc DEPS cinncore ARGS ${global_test_args})
cc_test(test_tensor SRCS tensor_test.cc DEPS cinncore)
cc_test(test_intrinsic_ops SRCS intrinsic_ops_test.cc DEPS cinncore)
cc_test(test_ir_verify SRCS ir_verify_test.cc DEPS cinncore)
