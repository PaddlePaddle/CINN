core_gather_headers()

core_gather_srcs(SRCS
    host_intrinsics.cc
    thread_backend.cc
)

if (WITH_MKL_BLAS)
    core_gather_srcs(SRCS mkl_math.cc cblas.cc)
endif()

if (WITH_MKLDNN)
    core_gather_srcs(SRCS mkldnn_math.cc )
endif()

if (WITH_MKL AND (NOT WITH_CUDA))
    cc_test(test_mkl_math SRCS mkl_math_test.cc mkl_math.cc DEPS cinncore)
endif()

if (WITH_MKLDNN)
    cc_test(test_mkldnn_math SRCS mkldnn_math_test.cc mkldnn_math.cc DEPS cinncore)
endif()

cc_test(test_host_intrinsics SRCS host_intrinsics_test.cc DEPS cinncore)
