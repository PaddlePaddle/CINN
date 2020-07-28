set(srcs op.h
        )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/hlir/${cpp}"
      CACHE INTERNAL "")
endforeach()

cc_test(test_op SRCS test_op.cc DEPS core)
