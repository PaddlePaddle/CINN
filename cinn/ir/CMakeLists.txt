set(srcs
    node.cc
    ir.cc
    node.cc
    ir_visitor.cc
    ir_printer.cc
    ir_mutator.cc
    function_definition.cc
    lowered_func.cc
    ir_operators.cc
    buffer.cc
    function_base.cc
    operation.cc
    collect_ir_nodes.cc
    registry.cc
    tensor.cc
    module.cc
    )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/ir/${cpp}"
      CACHE INTERNAL "")
endforeach()

message(STATUS "core_src: ${core_src}")

# cc_test(test_ir SRCS ir_test.cc DEPS core)
# cc_test(test_ir_printer SRCS ir_printer_test.cc DEPS core)
# cc_test(test_ir_operators SRCS ir_operators_test.cc DEPS core)
cc_test(test_collect_ir_nodes SRCS collect_ir_nodes_test.cc DEPS core)
# cc_test(test_tensor SRCS tensor_test.cc DEPS core)
cc_test(test_buffer SRCS buffer_test.cc DEPS core ARGS ${global_test_args})
cc_test(test_tensor SRCS tensor_test.cc DEPS core)
