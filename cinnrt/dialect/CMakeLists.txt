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
        )

mlir_tablegen_on(ops)
mlir_tablegen_on(basic_kernels)
mlir_tablegen_on(cinn_base DIALECT cinn)
mlir_tablegen_on(tensor_shape DIALECT ts)
mlir_tablegen_on(dense_tensor DIALECT dt)

# TODO(Superjomn) add a cmake function cc_executable to ecapsulate the following code
add_executable(cinn-opt opt.cc)
target_link_libraries(cinn-opt core ${mlir_libs})


# MLIR opt tests
# %{
add_test(test_mlir_opt_on_basic ${CMAKE_BINARY_DIR}/cinnrt/dialect/cinn-opt
        ${CMAKE_SOURCE_DIR}/cinn/dialect/mlir_tests/basic.mlir)

add_test(test_mlir_opt_on_tensor_shape ${CMAKE_BINARY_DIR}/cinnrt/dialect/cinn-opt
        ${CMAKE_SOURCE_DIR}/cinn/dialect/mlir_tests/tensor_shape.mlir)
# %}

cc_test(test_mlir_loader SRCS mlir_loader_test.cc DEPS core ${MLIR_IR_LIBS})

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinnrt/dialect/${cpp}"
      CACHE INTERNAL "")
endforeach()
