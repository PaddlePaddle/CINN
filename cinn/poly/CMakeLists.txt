set(srcs
    dim.cc
    domain.cc
    map.cc
    stage.cc
    isl_utils.cc
    schedule.cc
    naive_scheduler.cc
    poly_scheduler.cc
    ast_gen.cc
    graph.cc
    compute_at_transform.cc
)

set(includes
    ast_gen.h
    compute_at_transform.h
    dim.h
    domain.h
    graph.h
    isl_utils.h
    map.h
    naive_scheduler.h
    poly_scheduler.h
    schedule.h
    stage.h
    )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/poly/${cpp}"
      CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/poly/${header}" CACHE INTERNAL "")
endforeach()

# cc_test(test_graph SRCS graph_test.cc DEPS core)
cc_test(test_schedule SRCS schedule_test.cc DEPS cinncore)
# cc_test(test_poly_scheduler SRCS poly_scheduler_test.cc DEPS core)
cc_test(test_stage SRCS stage_test.cc DEPS cinncore)
# cc_test(test_ast_gen SRCS ast_gen_test.cc DEPS cinncore)
cc_test(test_compute_at_transform SRCS compute_at_transform_test.cc DEPS cinncore)
cc_test(test_ast_gen SRCS ast_gen_test.cc DEPS cinncore)
cc_test(test_isl_utils SRCS isl_utils_test.cc DEPS cinncore)
