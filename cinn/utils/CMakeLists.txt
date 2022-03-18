core_gather_headers()


gather_srcs(cinnapi_src SRCS
  dot_lang.cc
  error.cc
  functional.cc
  sized_multi_set.cc
  small_vector.cc
  string.cc
  timer.cc
  )

cc_test(test_string SRCS string_test.cc DEPS cinncore)
cc_test(test_sized_multi_set SRCS sized_multi_set_test.cc DEPS cinncore)
