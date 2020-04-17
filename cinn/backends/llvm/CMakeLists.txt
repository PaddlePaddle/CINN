add_definitions(${LLVM_DEFINITIONS})

find_package(LLVM REQUIRED CONFIG)
cc_library(llvm_util SRCS llvm_util.cc DEPS core LLVM)

cc_library(codegen_llvm SRCS codegen_llvm.cc DEPS llvm_util)

cc_library(simple_orc_jit SRCS simple_orc_jit.cc DEPS codegen_llvm)

cc_test(codegen_llvm_test SRCS codegen_llvm_test.cc DEPS codegen_llvm core)

cc_test(test_simple_orc_jit SRCS simple_orc_jit_test.cc DEPS simple_orc_jit core)
