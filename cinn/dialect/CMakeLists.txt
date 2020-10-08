set(srcs
        dialect.cc
        types.cc
        basic_kernels.cc
        cinn_base.cc
        init_cinn_dialects.cc
        )

# TODO(Superjomn) make function.
set(LLVM_TARGET_DEFINITIONS ops.td)
mlir_tablegen(ops.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
mlir_tablegen(ops.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
add_public_tablegen_target(OpsIncGen)
add_custom_target(OpsInc DEPENDS OpsIncGen)

set(LLVM_TARGET_DEFINITIONS basic_kernels.td)
mlir_tablegen(basic_kernels.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
mlir_tablegen(basic_kernels.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
add_public_tablegen_target(basic_kernels_inc_gen)
add_custom_target(basic_kernels_inc DEPENDS basic_kernels_inc_gen)

set(LLVM_TARGET_DEFINITIONS cinn_base.td)
mlir_tablegen(cinn_base.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
mlir_tablegen(cinn_base.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
add_public_tablegen_target(cinn_base_inc_gen)
add_custom_target(cinn_base_inc DEPENDS cinn_base_inc_gen)

add_executable(cinn_opt opt.cc)
target_link_libraries(cinn_opt core ${mlir_libs})

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/dialect/${cpp}"
      CACHE INTERNAL "")
endforeach()
