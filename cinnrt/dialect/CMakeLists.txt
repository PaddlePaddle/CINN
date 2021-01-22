set(srcs
        dialect.cc
        types.cc
        basic_kernels.cc
        cinn_base.cc
        init_cinn_dialects.cc
        tensor_shape.cc
        dense_tensor.cc
        mlir_loader.cc
        diagnostic_utils.cc
        pd_types.cc
        pd_ops.cc
        )

mlir_tablegen_on(ops)
mlir_tablegen_on(basic_kernels)
mlir_tablegen_on(cinn_base DIALECT cinn)
mlir_tablegen_on(tensor_shape DIALECT ts)
mlir_tablegen_on(dense_tensor DIALECT dt)
mlir_tablegen_on(pd_op_base DIALECT pd)
mlir_tablegen_on(pd_ops)

# TODO(Superjomn) add a cmake function cc_executable to ecapsulate the following code
add_executable(cinn-opt opt.cc)
target_link_libraries(cinn-opt cinncore ${mlir_libs})
add_dependencies(cinn-opt pd_ops_inc)
add_executable(print-ir print_ir.cc)
target_link_libraries(print-ir cinncore ${mlir_libs})
add_dependencies(print-ir pd_ops_inc)


# MLIR opt tests
# %{
add_test(test_mlir_opt_on_basic ${CMAKE_BINARY_DIR}/cinnrt/dialect/cinn-opt
        ${CMAKE_SOURCE_DIR}/cinnrt/dialect/mlir_tests/basic.mlir)

set(cinn_opt_path ${CMAKE_BINARY_DIR}/cinnrt/dialect/cinn-opt)
add_test(test_mlir_opt_on_tensor_shape ${cinn_opt_path}
        ${CMAKE_SOURCE_DIR}/cinnrt/dialect/mlir_tests/tensor_shape.mlir)
add_test(test_mlir_opt_on_paddle_ops
        ${cinn_opt_path}
        ${CMAKE_SOURCE_DIR}/cinnrt/dialect/mlir_tests/paddle_ops.mlir)
# %}

cc_test(test_mlir_loader SRCS mlir_loader_test.cc DEPS cinncore ${MLIR_IR_LIBS})

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinnrt/dialect/${cpp}"
      CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
