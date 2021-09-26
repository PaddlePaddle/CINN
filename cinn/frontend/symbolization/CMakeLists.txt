core_gather_headers()
core_gather_srcs(SRCS
    base_builder.cc
    net_builder.cc
    )

if(NOT WITH_CUDA)
  cc_test(test_net_builder
          SRCS net_builder_test.cc DEPS cinncore)
else()
  nv_test(test_net_builder
          SRCS net_builder_test.cc DEPS cinncore )
endif()
