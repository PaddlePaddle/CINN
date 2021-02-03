core_gather_headers()
core_gather_srcs(SRCS
  syntax.cc
  paddle_model_to_program.cc
  interpreter.cc
  )

if(NOT WITH_CUDA)
  cc_test(test_frontend_syntax
          ARGS "--model_dir=${THIRD_PARTY_PATH}/naive_mul_model"
          SRCS syntax_test.cc DEPS cinncore)

  cc_test(test_frontend_interpreter
          ARGS --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
          SRCS interpreter_test.cc DEPS cinncore)
else()
  nv_test(test_frontend_syntax
          ARGS "--model_dir=${THIRD_PARTY_PATH}/naive_mul_model"
          SRCS syntax_test.cc DEPS cinncore )

  nv_test(test_frontend_interpreter
          ARGS --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
          SRCS interpreter_test.cc DEPS cinncore)
endif()


add_subdirectory(paddle)

