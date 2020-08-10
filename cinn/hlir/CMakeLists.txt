add_subdirectory(framework)
add_subdirectory(instruction)
add_subdirectory(pe)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/hlir/${cpp}"
      CACHE INTERNAL "")
endforeach()

cc_test(test_op SRCS op_test.cc DEPS core)
cc_test(test_print_graph_pass SRCS print_graph_pass_test.cc DEPS core)
