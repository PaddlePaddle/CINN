set(srcs
        dialect.cc
        types.cc
        )

set(LLVM_TARGET_DEFINITIONS ops.td)
mlir_tablegen(ops.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
mlir_tablegen(ops.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
add_public_tablegen_target(OpsIncGen)

set(LLVM_TARGET_DEFINITIONS basic_kernels.td)
mlir_tablegen(basic_kernels.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
mlir_tablegen(basic_kernels.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
add_public_tablegen_target(basic_kernels_inc_gen)

add_custom_target(OpsInc DEPENDS OpsIncGen)
add_custom_target(basic_kernels_inc DEPENDS basic_kernels_inc_gen)


foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/hlir/dialect/${cpp}"
      CACHE INTERNAL "")
endforeach()
