set(srcs
        kernel_frame.cc
        kernel_registry.cc
        value.cc
        kernel_utils.cc
        symbol_table.cc
        op_executable.cc
        core_runtime.cc
        tensor_shape.cc
        mlir_to_runtime_translate.cc
  )


cc_test(test_host_context_value SRCS value_test.cc DEPS core)
cc_test(test_kernel_utils SRCS kernel_utils_test.cc DEPS core)
cc_test(test_kernel_registry SRCS kernel_registry_test.cc DEPS core)
cc_test(test_op_executable SRCS op_executable_test.cc DEPS core)
cc_test(test_core_runtime SRCS core_runtime_test.cc DEPS core)
cc_test(test_mlir_to_runtime_translate SRCS mlir_to_runtime_translate_test.cc DEPS core ${MLIR_IR_LIBS})

add_test(test_mlir_exec_on_basic ${CMAKE_BINARY_DIR}/cinn/host_context/cinn-exec -i
        ${CMAKE_CURRENT_SOURCE_DIR}/mlir_tests/basic.mlir | FileChecker  ${CMAKE_CURRENT_SOURCE_DIR}/mlir_tests/basic.mlir)
add_test(test_mlir_exec_on_basic ${CMAKE_BINARY_DIR}/cinn/host_context/cinn-exec -i
        ${CMAKE_CURRENT_SOURCE_DIR}/mlir_tests/shape.mlir | FileChecker  ${CMAKE_CURRENT_SOURCE_DIR}/mlir_tests/shape.mlir)

add_executable(cinn-exec mlir_exec.cc)
target_link_libraries(cinn-exec core ${MLIR_IR_LIBS})

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/host_context/${cpp}"
    CACHE INTERNAL "")
endforeach()
