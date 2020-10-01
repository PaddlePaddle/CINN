set(srcs string.cc functional.cc dot_lang.cc timer.cc llvm_utils.cc)

cc_test(test_string SRCS string_test.cc DEPS core)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/utils/${cpp}"
      CACHE INTERNAL "")
endforeach()
