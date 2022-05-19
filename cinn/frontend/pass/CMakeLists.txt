core_gather_headers()

gather_srcs(cinnapi_src SRCS
    decomposer.cc
    dead_code_eliminate.cc
    remove_identity.cc
    transpose_collapsing.cc
    transpose_folding_input.cc
    transpose_folding_output.cc
    gemm_rewriter.cc
    reshape_rewriter.cc
    fill_constant_folding.cc
    matmul_merge.cc
    )


cc_test(test_decomposer_pass SRCS decomposer_test.cc DEPS cinncore)
cc_test(test_dead_code_eliminate_pass SRCS dead_code_eliminate_test.cc DEPS cinncore)
cc_test(test_remove_identity_pass SRCS remove_identity_test.cc DEPS cinncore)
cc_test(test_transpose_collapsing SRCS transpose_collapsing_test.cc DEPS cinncore)
cc_test(test_transpose_folding_input_pass SRCS transpose_folding_input_test.cc DEPS cinncore)
cc_test(test_gemm_rewriter_pass SRCS gemm_rewriter_test.cc DEPS cinncore)
cc_test(test_transpose_folding_output_pass SRCS transpose_folding_output_test.cc DEPS cinncore)
cc_test(test_reshape_rewriter_pass SRCS reshape_rewriter_test.cc DEPS cinncore)
cc_test(test_fill_constant_folding_pass SRCS fill_constant_folding_test.cc DEPS cinncore)
