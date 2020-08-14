set(srcs
  syntax.cc
  api.cc
  )

cc_test(test_frontend_syntax SRCS syntax_test.cc DEPS core)


foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/frontend/${cpp}"
    CACHE INTERNAL "")
endforeach()
