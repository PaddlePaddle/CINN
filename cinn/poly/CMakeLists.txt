cc_library(poly SRCS
        dim.cc
        domain.cc
        map.cc
        stage.cc
        isl_utils.cc
        schedule.cc
        ast_gen.cc
        graph.cc
        DEPS common ir)

cc_test(test_stage SRCS stage_test.cc DEPS poly)
cc_test(test_schedule SRCS schedule_test.cc DEPS poly)
cc_test(test_ast_gen SRCS ast_gen_test.cc DEPS poly)
cc_test(test_graph SRCS graph_test.cc DEPS poly)
