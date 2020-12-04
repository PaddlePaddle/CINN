set(srcs
  syntax.cc
  paddle_model_to_program.cc
  interpreter.cc
  )

set(includes
    interpreter.h
    paddle_model_to_program.h
    syntax.h
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

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/frontend/${cpp}"
    CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/frontend/${header}" CACHE INTERNAL "")
endforeach()
