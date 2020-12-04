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
  )

set(includes
    buffer.h
    graph.h
    graph_compiler.h
    instruction.h
    memory.h
    node.h
    op.h
    op_strategy.h
    pass.h
    schedule.h
    scope.h
    tensor.h
    variable.h
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

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/hlir/framework/${cpp}"
    CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/hlir/framework/${header}" CACHE INTERNAL "")
endforeach()
