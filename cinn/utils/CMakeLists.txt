core_gather_headers()

core_gather_srcs(SRCS
  string.cc
  functional.cc
  dot_lang.cc
  timer.cc
  error.cc
  small_vector.cc
  )

cc_test(test_string SRCS string_test.cc DEPS cinncore)
