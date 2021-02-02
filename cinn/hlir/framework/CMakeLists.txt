core_gather_headers()

core_gather_srcs(
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
  )

if(WITH_CUDA)
  nv_test(test_hlir_framework_buffer SRCS buffer_test.cc DEPS cinncore)
  nv_test(test_hlir_framework_infershape_pass SRCS infershape_pass_test.cc DEPS cinncore)
  nv_test(test_cuda_graph_compiler SRCS cuda_graph_compiler_test.cc DEPS cinncore)
else()
  cc_test(test_hlir_framework_buffer SRCS buffer_test.cc DEPS cinncore)
  cc_test(test_hlir_framework_infershape_pass SRCS infershape_pass_test.cc DEPS cinncore)
endif()

cc_test(test_hlir_framework_tensor SRCS tensor_test.cc DEPS cinncore)
cc_test(test_hlir_framework_scope SRCS scope_test.cc DEPS cinncore)
cc_test(test_hlir_framework_instruction SRCS instruction_test.cc DEPS cinncore)
cc_test(test_hlir_framework_op SRCS op_test.cc DEPS cinncore)
cc_test(test_hlir_framework_print_graph_pass SRCS print_graph_pass_test.cc DEPS cinncore)
