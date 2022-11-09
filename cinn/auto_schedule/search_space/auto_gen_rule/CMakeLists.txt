core_gather_headers()

gather_srcs(cinnapi_src SRCS
	auto_gen_rule.cc
	auto_inline.cc
	auto_unroll.cc
	multi_level_tiling.cc
	skip_rule.cc
  add_cache_read.cc
	)

cc_test(test_auto_inline SRCS auto_inline_test.cc DEPS cinncore)
cc_test(test_multi_level_tiling SRCS multi_level_tiling_test.cc DEPS cinncore)
cc_test(test_skip_rule SRCS skip_rule_test.cc DEPS cinncore)
cc_test(test_auto_unroll SRCS auto_unroll_test.cc DEPS cinncore)
cc_test(test_add_cache_read SRCS add_cache_read_test.cc DEPS cinncore)
