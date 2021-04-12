core_gather_headers()

core_gather_srcs(SRCS
    cinnrt_api.cc
    )

add_executable(predictor predictor.cc)
target_link_libraries(predictor cinncore ${MLIR_IR_LIBS})
