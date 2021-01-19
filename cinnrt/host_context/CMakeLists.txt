set(srcs
        kernel_frame.cc
        kernel_registry.cc
        value.cc
        kernel_utils.cc
        symbol_table.cc
        op_executable.cc
        core_runtime.cc
        tensor_shape.cc
        dense_host_tensor.cc
        dense_tensor_view.cc
        mlir_to_runtime_translate.cc
        tensor_metadata.cc
  )

cc_test(test_host_context_value SRCS value_test.cc DEPS cinncore)
cc_test(test_kernel_utils SRCS kernel_utils_test.cc DEPS cinncore)
cc_test(test_kernel_registry SRCS kernel_registry_test.cc DEPS cinncore)
cc_test(test_op_executable SRCS op_executable_test.cc DEPS cinncore)
cc_test(test_core_runtime SRCS core_runtime_test.cc DEPS cinncore)
cc_test(test_mlir_to_runtime_translate SRCS mlir_to_runtime_translate_test.cc DEPS cinncore ${MLIR_IR_LIBS})

cinn_exec_check(test_mlir_exec_on_basic mlir_tests/basic.mlir)
cinn_exec_check(test_mlir_exec_on_shape mlir_tests/shape.mlir)
cinn_exec_check(test_mlir_exec_on_dense_tensor mlir_tests/dense_tensor.mlir)

add_executable(cinn-exec mlir_exec.cc)
target_link_libraries(cinn-exec cinncore ${MLIR_IR_LIBS})

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinnrt/host_context/${cpp}"
    CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
