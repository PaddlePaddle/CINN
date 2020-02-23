cc_library(poly SRCS
        dim.cc
        domain.cc
        map.cc
        stage.cc
        isl_utils.cc
        schedule.cc
        ast_gen.cc
        DEPS common ir)

cc_test(test_poly_element SRCS stage_test.cc DEPS poly)
cc_test(test_schedule SRCS schedule_test.cc DEPS poly)
cc_test(test_ast_gen SRCS ast_gen_test.cc DEPS poly)
