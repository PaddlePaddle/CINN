set(srcs
    buffer.cc
    compute.cc
    placeholder.cc
    lower.cc
    builtin.cc
    lower_impl.cc
    compute_at_postprocess.cc
    packed_func.cc
    )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/lang/${cpp}"
      CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()

cc_test(test_compute SRCS compute_test.cc DEPS cinncore)
cc_test(test_placeholder SRCS placeholder_test.cc DEPS cinncore)
cc_test(test_lower SRCS lower_test.cc DEPS cinncore)
cc_test(test_lower_impl SRCS lower_impl_test.cc DEPS cinncore)
cc_test(test_packed_func SRCS packed_func_test.cc DEPS cinncore)
