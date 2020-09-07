set(srcs
  syntax.cc
  )

if(NOT WITH_CUDA)
cc_test(test_frontend_syntax SRCS syntax_test.cc DEPS core)
else()
nv_test(test_frontend_syntax SRCS syntax_test.cc DEPS core)
endif()


add_subdirectory(paddle)

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/frontend/${cpp}"
    CACHE INTERNAL "")
endforeach()
