set(srcs outputs.cc codegen_c.cc)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/backends/${cpp}"
      CACHE INTERNAL "")
endforeach()


cc_test(test_codegen_c SRCS codegen_c_test.cc DEPS core)
