core_gather_headers()

gather_srcs(cinnapi_src SRCS
    decomposer.cc
    remove_identity.cc
    transpose_folding.cc
    gemm_rewriter.cc
    fill_constant_folding.cc
    )


cc_test(test_decomposer_pass SRCS decomposer_test.cc DEPS cinncore)
cc_test(test_remove_identity_pass SRCS remove_identity_test.cc DEPS cinncore)
cc_test(test_transpose_folding_pass SRCS transpose_folding_test.cc DEPS cinncore)
cc_test(test_gemm_rewriter_pass SRCS gemm_rewriter_test.cc DEPS cinncore)
cc_test(test_fill_constant_folding_pass SRCS fill_constant_folding_test.cc DEPS cinncore)
