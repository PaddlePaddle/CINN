set(srcs
    dim.cc
    domain.cc
    map.cc
    stage.cc
    isl_utils.cc
    schedule.cc
    ast_gen.cc
    graph.cc
  )


foreach(cpp ${srcs})
  set (core_src "${core_src};cinn/poly/${cpp}" CACHE INTERNAL "")
endforeach()


cc_test(test_graph SRCS graph_test.cc DEPS core)
cc_test(test_schedule SRCS schedule_test.cc DEPS core)
cc_test(test_stage SRCS stage_test.cc DEPS core)
cc_test(test_ast_gen SRCS ast_gen_test.cc DEPS core)
