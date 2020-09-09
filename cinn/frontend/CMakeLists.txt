set(srcs
  syntax.cc
  )

if(NOT WITH_CUDA)
  cc_test(test_frontend_syntax SRCS syntax_test.cc DEPS core
    ARGS --model_dir=${THIRD_PARTY_PATH}/model/lite_naive_model)
else()
  nv_test(test_frontend_syntax SRCS syntax_test.cc DEPS core
    ARGS --model_dir=${THIRD_PARTY_PATH}/model/lite_naive_model)
endif()


add_subdirectory(paddle)

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/frontend/${cpp}"
    CACHE INTERNAL "")
endforeach()
