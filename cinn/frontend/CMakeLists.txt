core_gather_headers()
gather_srcs(cinnapi_src SRCS
  syntax.cc
  paddle_model_to_program.cc
  interpreter.cc
  base_builder.cc
  net_builder.cc
  cinn_builder.cc
  paddle_model_to_netbuilder.cc)

if(NOT WITH_CUDA)
  cc_test(test_frontend_syntax
          ARGS "--model_dir=${THIRD_PARTY_PATH}/naive_mul_model"
          SRCS syntax_test.cc DEPS cinncore)

  cc_test(test_frontend_interpreter
          ARGS --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
          SRCS interpreter_test.cc DEPS cinncore)

  cc_test(test_paddle_model_to_netbuilder
          ARGS --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
          SRCS paddle_model_to_netbuilder_test.cc DEPS cinncore)
else()
  nv_test(test_frontend_syntax
          ARGS "--model_dir=${THIRD_PARTY_PATH}/naive_mul_model"
          SRCS syntax_test.cc DEPS cinncore )

  nv_test(test_frontend_interpreter
          ARGS --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
          SRCS interpreter_test.cc DEPS cinncore)

  nv_test(test_paddle_model_to_netbuilder
          ARGS --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
          SRCS paddle_model_to_netbuilder_test.cc DEPS cinncore)
endif()

cc_test(test_net_builder SRCS net_builder_test.cc DEPS cinncore)
cc_test(test_cinn_builder SRCS cinn_builder_test.cc DEPS cinncore)
cc_test(test_decomposer_registry
        SRCS decomposer_registry_test.cc DEPS cinncore)

add_subdirectory(paddle)
add_subdirectory(decomposer)
add_subdirectory(op_mappers)

cc_test(test_op_mapper_registry SRCS op_mapper_registry_test.cc DEPS cinncore)
