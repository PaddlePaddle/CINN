set(srcs
  syntax.cc
  paddle_model_to_program.cc
  )

if(NOT WITH_CUDA)
  cc_test(test_frontend_syntax
          ARGS --model_dir=${THIRD_PARTY_PATH}/model/lite_naive_model
          SRCS syntax_test.cc DEPS core)
else()
  nv_test(test_frontend_syntax
          ARGS --model_dir=${THIRD_PARTY_PATH}/model/lite_naive_model
          SRCS syntax_test.cc
          DEPS core
          )
endif()


add_subdirectory(paddle)

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/frontend/${cpp}"
    CACHE INTERNAL "")
endforeach()
