set(srcs
    node.cc
    ir.cc
    node.cc
    ir_visitor.cc
    ir_printer.cc
    ir_mutator.cc
    function.cc
    function_definition.cc
    ir_operators.cc
    buffer.cc
    function_base.cc
    operation.cc)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/ir/${cpp}"
      CACHE INTERNAL "")
endforeach()

message(STATUS "core_src: ${core_src}")

cc_test(test_ir SRCS ir_test.cc DEPS core)
cc_test(test_ir_printer SRCS ir_printer_test.cc DEPS core)
cc_test(test_ir_operators SRCS ir_operators_test.cc DEPS core)
# cc_test(test_tensor SRCS tensor_test.cc DEPS core)
cc_test(test_function SRCS function_test.cc DEPS core)
cc_test(test_buffer SRCS buffer_test.cc DEPS core)
