add_definitions(${LLVM_DEFINITIONS})

# generate cinn_runtime.ll file
execute_process(
    COMMAND clang++ -mavx2 -masm=intel -S -emit-llvm -O3 ${PROJECT_SOURCE_DIR}/cinn/runtime/cinn_runtime.cc -I${PROJECT_SOURCE_DIR}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cinn/backends/llvm
)

execute_process(
    COMMAND cat cinn_runtime.ll
    COMMAND python3 generate_runtime_llvm_ir.py
    OUTPUT_FILE cinn_runtime_llvm_ir.h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cinn/backends/llvm
)

cc_library(llvm_util SRCS llvm_util.cc DEPS core LLVM)

cc_library(codegen_llvm SRCS codegen_llvm.cc DEPS llvm_util)

cc_library(simple_orc_jit SRCS simple_orc_jit.cc DEPS codegen_llvm)

cc_test(test_codegen_llvm SRCS codegen_llvm_test.cc DEPS codegen_llvm core)

cc_test(test_simple_orc_jit SRCS simple_orc_jit_test.cc DEPS simple_orc_jit core)
