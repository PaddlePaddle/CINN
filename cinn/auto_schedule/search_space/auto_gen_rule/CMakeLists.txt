core_gather_headers()

gather_srcs(cinnapi_src SRCS
	auto_gen_rule.cc
	auto_inline.cc
	auto_unroll.cc
	multi_level_tiling.cc
	skip_rule.cc
    add_cache_read.cc
    add_cache_write.cc
    auto_bind.cc
	)

if (WITH_TESTING)
  cc_library(auto_gen_rule_test_helper SRCS test_helper.cc DEPS glog gtest cinncore)
endif()

if (WITH_CUDA)
    nv_test(test_add_cache_read SRCS add_cache_read_test.cc DEPS cinncore auto_gen_rule_test_helper)
    nv_test(test_add_cache_write SRCS add_cache_write_test.cc DEPS cinncore auto_gen_rule_test_helper)
    nv_test(test_mix_rules SRCS mix_rules_test.cc DEPS cinncore auto_gen_rule_test_helper)
    nv_test(test_auto_bind SRCS auto_bind_test.cc DEPS cinncore auto_gen_rule_test_helper)
endif()

cc_test(test_auto_inline SRCS auto_inline_test.cc DEPS cinncore)
cc_test(test_multi_level_tiling SRCS multi_level_tiling_test.cc DEPS cinncore)
cc_test(test_skip_rule SRCS skip_rule_test.cc DEPS cinncore)
cc_test(test_auto_unroll SRCS auto_unroll_test.cc DEPS cinncore)
