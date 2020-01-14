cc_library(ir SRCS
        type.cc node.cc ir.cc node.cc
        ir_visitor.cc
        ir_printer.cc
        )

cc_test(test_ir SRCS ir_test.cc DEPS ir)
