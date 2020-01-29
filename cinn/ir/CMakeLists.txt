cc_library(ir SRCS
        node.cc ir.cc node.cc
        ir_visitor.cc
        ir_printer.cc
        function.cc
        function_definition.cc
        ir_operators.cc
        buffer.cc
        tensor.cc
        function_base.cc
        operation.cc
        DEPS common boost
        )

cc_test(test_ir SRCS ir_test.cc DEPS ir)
cc_test(test_ir_printer SRCS ir_printer_test.cc DEPS ir)
cc_test(test_ir_operators SRCS ir_operators_test.cc DEPS ir)
cc_test(test_tensor SRCS tensor_test.cc DEPS ir)
cc_test(test_function SRCS function_test.cc DEPS ir)
