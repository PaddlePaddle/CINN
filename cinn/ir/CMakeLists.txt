cc_library(ir SRCS
        node.cc ir.cc node.cc
        ir_visitor.cc
        ir_printer.cc
        function.cc
        function_definition.cc
        ir_operators.cc
        DEPS common boost
        )

cc_test(test_ir SRCS ir_test.cc DEPS ir)
cc_test(test_ir_printer SRCS ir_printer_test.cc DEPS ir)
cc_test(test_ir_operators SRCS ir_operators_test.cc DEPS ir)
