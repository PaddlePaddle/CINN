core_gather_headers()

core_gather_srcs(SRCS
    cinnrt_api.cc
    )

cc_test(test_cinnrt_api SRCS cinnrt_api_test.cc DEPS cinncore ${MLIR_IR_LIBS})
