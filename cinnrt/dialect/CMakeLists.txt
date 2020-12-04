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

set(includes
    basic_kernels.h
    cinn_base.h
    dense_tensor.h
    diagnostic_utils.h
    init_cinn_dialects.h
    mlir_loader.h
    tensor_shape.h
    types.h
    )

mlir_tablegen_on(ops)
mlir_tablegen_on(basic_kernels)
mlir_tablegen_on(cinn_base DIALECT cinn)
mlir_tablegen_on(tensor_shape DIALECT ts)
mlir_tablegen_on(dense_tensor DIALECT dt)

# TODO(Superjomn) add a cmake function cc_executable to ecapsulate the following code
add_executable(cinn-opt opt.cc)
target_link_libraries(cinn-opt cinncore ${mlir_libs})


# MLIR opt tests
# %{
add_test(test_mlir_opt_on_basic ${CMAKE_BINARY_DIR}/cinnrt/dialect/cinn-opt
        ${CMAKE_SOURCE_DIR}/cinnrt/dialect/mlir_tests/basic.mlir)

add_test(test_mlir_opt_on_tensor_shape ${CMAKE_BINARY_DIR}/cinnrt/dialect/cinn-opt
        ${CMAKE_SOURCE_DIR}/cinnrt/dialect/mlir_tests/tensor_shape.mlir)
# %}

cc_test(test_mlir_loader SRCS mlir_loader_test.cc DEPS cinncore ${MLIR_IR_LIBS})

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinnrt/dialect/${cpp}"
      CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinnrt/dialect/${header}" CACHE INTERNAL "")
endforeach()
