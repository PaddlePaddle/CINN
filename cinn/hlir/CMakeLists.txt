add_subdirectory(instruction)
set(srcs op.h
        )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/hlir/${cpp}"
      CACHE INTERNAL "")
endforeach()

cc_test(test_op SRCS op_test.cc DEPS core)
