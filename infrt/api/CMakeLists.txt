core_gather_headers()

gather_srcs(infrt_src SRCS
    infrt_api.cc
    )

cc_test(test_infrt_api SRCS infrt_api_test.cc DEPS infrt ${MLIR_IR_LIBS})
