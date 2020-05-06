add_definitions(${LLVM_DEFINITIONS})

# generate cinn_runtime.ll file

add_custom_target(GEN_LLVM_RUNTIME_IR
    ALL
    COMMAND clang++ -mavx2 -std=c++11 -masm=intel -S -emit-llvm -O3 ${PROJECT_SOURCE_DIR}/cinn/runtime/cinn_runtime.cc -I${PROJECT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(GEN_LLVM_RUNTIME_IR_HEADER
        ALL
        COMMAND python3 generate_runtime_llvm_ir.py ${CMAKE_BINARY_DIR}/cinn_runtime.ll
            ${CMAKE_BINARY_DIR}/cinn/backends/llvm/cinn_runtime_llvm_ir.h
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cinn/backends/llvm
        )
add_dependencies(GEN_LLVM_RUNTIME_IR_HEADER GEN_LLVM_RUNTIME_IR)

cc_library(llvm_util SRCS llvm_util.cc DEPS core LLVM)
add_dependencies(llvm_util GEN_LLVM_RUNTIME_IR_HEADER)

cc_library(runtime_registry SRCS runtime_registry.cc)

cc_library(codegen_llvm SRCS codegen_llvm.cc DEPS llvm_util)

cc_library(simple_orc_jit SRCS simple_orc_jit.cc DEPS codegen_llvm runtime_registry)

cc_test(test_codegen_llvm SRCS codegen_llvm_test.cc DEPS codegen_llvm core)

cc_test(test_simple_orc_jit SRCS simple_orc_jit_test.cc DEPS simple_orc_jit core)
