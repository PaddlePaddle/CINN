set(srcs
  syntax.cc
  )

foreach(cpp ${srcs})
  set(core_src "${core_src};cinn/frontend/${cpp}" CACHE INTERNAL "")
endforeach()

add_subdirectory(paddle)

cc_test(test_frontend_syntax SRCS syntax_test.cc DEPS core)
