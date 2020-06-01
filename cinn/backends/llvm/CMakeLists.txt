add_definitions(${LLVM_DEFINITIONS})

# generate cinn_runtime.ll file

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/cinn/backends/llvm/cinn_runtime_llvm_ir.h
  COMMAND clang++ -mavx2 -std=c++11 -masm=intel -S -emit-llvm -O3 ${PROJECT_SOURCE_DIR}/cinn/runtime/cinn_runtime.cc -I${PROJECT_SOURCE_DIR}
  COMMAND python3 generate_runtime_llvm_ir.py ${PROJECT_SOURCE_DIR}/cinn/backends/llvm/cinn_runtime.ll ${CMAKE_BINARY_DIR}/cinn/backends/llvm/cinn_runtime_llvm_ir.h
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cinn/backends/llvm
  DEPENDS ${PROJECT_SOURCE_DIR}/cinn/runtime/cinn_runtime.cc ${PROJECT_SOURCE_DIR}/cinn/runtime/cinn_runtime.h
  )

add_custom_target(GEN_LLVM_RUNTIME_IR_HEADER ALL
  DEPENDS ${CMAKE_BINARY_DIR}/cinn/backends/llvm/cinn_runtime_llvm_ir.h
  )

set(srcs
  llvm_util.cc
  runtime_symbol_registry.cc
  codegen_llvm.cc
  simple_orc_jit.cc
  )

cc_test(test_codegen_llvm SRCS codegen_llvm_test.cc DEPS core)

cc_test(test_simple_orc_jit SRCS simple_orc_jit_test.cc DEPS core)


foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/backends/llvm/${cpp}"
    CACHE INTERNAL "")
endforeach()
