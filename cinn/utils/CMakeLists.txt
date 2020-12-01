set(srcs string.cc functional.cc dot_lang.cc timer.cc error.cc
        small_vector.cc)

cc_test(test_string SRCS string_test.cc DEPS cinncore)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/utils/${cpp}"
      CACHE INTERNAL "")
endforeach()
