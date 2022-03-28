core_gather_headers()

gather_srcs(cinnapi_src SRCS
	auto_gen_rule.cc
	auto_inline.cc
	multi_level_tiling.cc
	)


#cc_test(test_multi_level_tiling SRCS multi_level_tiling.cc DEPS cinncore)
#cc_test(test_auto_inline SRCS auto_inline_test.cc DEPS cinncore)
