core_gather_headers()

core_gather_srcs(SRCS
    dialect.cc
    types.cc
    basic_kernels.cc
    test_kernels.cc
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
mlir_tablegen_on(test_kernels)
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
set(cinn_opt_path ${CMAKE_BINARY_DIR}/cinnrt/dialect/cinn-opt)

add_test(test_mlir_opt_on_basic ${cinn_opt_path}
        ${CMAKE_SOURCE_DIR}/cinnrt/dialect/mlir_tests/basic.mlir)
add_test(test_mlir_opt_on_tensor_shape ${cinn_opt_path}
        ${CMAKE_SOURCE_DIR}/cinnrt/dialect/mlir_tests/tensor_shape.mlir)
add_test(test_mlir_opt_on_paddle_ops
        ${cinn_opt_path}
        ${CMAKE_SOURCE_DIR}/cinnrt/dialect/mlir_tests/paddle_ops.mlir)
# %}

cc_test(test_mlir_loader SRCS mlir_loader_test.cc DEPS cinncore ${MLIR_IR_LIBS})

# execute mlir and run FileCheck
cinn_exec_check(run_and_check_tensor_type mlir_tests/tensor_type.mlir)
cinn_exec_check(run_and_check_basic mlir_tests/basic.mlir)
cinn_exec_check(run_and_check_benchmark mlir_tests/benchmark.mlir)
cinn_exec_check(run_and_check_return_tensor_pointer mlir_tests/return_tensor_pointer.mlir)
#cinn_exec_check(run_and_check_dense_tensor mlir_tests/dense_tensor.mlir)
add_test(test_mlir_dense_tensor
        ${CMAKE_BINARY_DIR}/cinnrt/host_context/cinn-exec
        -i
        ${CMAKE_CURRENT_SOURCE_DIR}/mlir_tests/dense_tensor.mlir)
