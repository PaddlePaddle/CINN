core_gather_headers()
core_gather_srcs(SRCS
    base_builder.cc
    coarse_builder.cc
    )

if(NOT WITH_CUDA)
  cc_test(test_coarse_builder
          SRCS coarse_builder_test.cc DEPS cinncore)
else()
  nv_test(test_coarse_builder
          SRCS coarse_builder_test.cc DEPS cinncore )
endif()
