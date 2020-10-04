set(srcs
  tensor.cc
  scope.cc
  variable.cc
  buffer.cc
  memory.cc
  instruction.cc
  graph_compiler.cc
  graph.cc
  node.cc
  pass.cc
  op_strategy.cc
  kernel_registry.cc
  kernel_utils.cc
  )

if(WITH_CUDA)
  nv_test(test_hlir_framework_buffer SRCS buffer_test.cc DEPS core)
  nv_test(test_hlir_framework_infershape_pass SRCS infershape_pass_test.cc DEPS core)
else()
  cc_test(test_hlir_framework_buffer SRCS buffer_test.cc DEPS core)
  cc_test(test_hlir_framework_infershape_pass SRCS infershape_pass_test.cc DEPS core)
endif()

cc_test(test_hlir_framework_tensor SRCS tensor_test.cc DEPS core)
cc_test(test_hlir_framework_scope SRCS scope_test.cc DEPS core)
cc_test(test_hlir_framework_instruction SRCS instruction_test.cc DEPS core)
cc_test(test_hlir_framework_op SRCS op_test.cc DEPS core)
cc_test(test_hlir_framework_print_graph_pass SRCS print_graph_pass_test.cc DEPS core)

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/hlir/framework/${cpp}"
    CACHE INTERNAL "")
endforeach()
