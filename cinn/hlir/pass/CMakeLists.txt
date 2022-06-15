core_gather_headers()


gather_srcs(cinnapi_src SRCS
    infershape.cc
    opfusion.cc
    alterlayout.cc
    const_propagate.cc
    op_fusion_pass.cc
    fusion_merge_pass.cc
    dot_merger.cc
    )

cc_test(test_opfusion SRCS opfusion_test.cc DEPS cinncore)
cc_test(test_primitive_ops SRCS test_primitive_ops.cc DEPS cinncore)
cc_test(test_op_fusion_pass SRCS op_fusion_pass_test.cc DEPS cinncore)
cc_test(test_fusion_merge_pass SRCS fusion_merge_pass_test.cc DEPS cinncore)
if (NOT WITH_CUDA)
cc_test(test_alterlayout SRCS alterlayout_test.cc DEPS cinncore)
endif()
cc_test(test_const_propagate SRCS const_propagate_test.cc DEPS cinncore)
