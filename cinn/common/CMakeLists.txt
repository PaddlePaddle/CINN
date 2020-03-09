set(srcs
    shared.cc
    pod_value.cc
    type.cc
    target.cc
    object.cc
    graph_utils.cc
    context.cc
    axis.cc
    ir.cc
    )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/common/${cpp}"
      CACHE INTERNAL "")
endforeach()

cc_test(test_pod_value SRCS pod_value_test.cc DEPS core)
cc_test(test_shared SRCS shared_test.cc DEPS core)
cc_test(test_graph_utils SRCS graph_utils_test.cc DEPS core)
