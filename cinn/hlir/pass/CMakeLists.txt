core_gather_headers()

core_gather_srcs(SRCS
    infershape.cc
    opfusion.cc
    alterlayout.cc
    )

if(NOT WITH_CUDA)
    cc_test(test_alterlayout SRCS alterlayout_test.cc DEPS cinncore)
endif()
