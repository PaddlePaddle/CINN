core_gather_headers()

core_gather_srcs(SRCS
    infershape.cc
    opfusion.cc
    alterlayout.cc
    )


cc_test(test_opfusion SRCS opfusion_test.cc DEPS cinncore)
cc_test(test_primitive_ops SRCS primitive_ops_demo.cc DEPS cinncore)
if (NOT WITH_CUDA)
cc_test(test_alterlayout SRCS alterlayout_test.cc DEPS cinncore)
endif()
