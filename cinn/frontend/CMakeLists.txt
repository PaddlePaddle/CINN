set(srcs
  syntax.cc
  paddle_model_to_program.cc
  interpreter.cc
  )

if(NOT WITH_CUDA)
  cc_test(test_frontend_syntax
          ARGS "--model_dir=${THIRD_PARTY_PATH}/naive_mul_model"
          SRCS syntax_test.cc DEPS core)

  cc_test(test_frontend_interpreter
          ARGS --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
          SRCS interpreter_test.cc DEPS core)
else()
  nv_test(test_frontend_syntax
          ARGS "--model_dir=${THIRD_PARTY_PATH}/naive_mul_model"
          SRCS syntax_test.cc DEPS core )

  nv_test(test_frontend_interpreter
          ARGS --model_dir=${THIRD_PARTY_PATH}/naive_mul_model
          SRCS interpreter_test.cc DEPS core)
endif()


add_subdirectory(paddle)

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/frontend/${cpp}"
    CACHE INTERNAL "")
endforeach()
