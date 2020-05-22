set(srcs
    shared.cc
    cinn_value.cc
    type.cc
    target.cc
    object.cc
    debug_manager.cc
    info_registry.cc
    graph_utils.cc
    context.cc
    axis.cc
    ir_util.cc
    test_helper.cc
    arithmatic.cc
    cas.cc
    )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/common/${cpp}"
      CACHE INTERNAL "")
endforeach()

cc_test(test_cinn_value SRCS cinn_value_test.cc DEPS core)
cc_test(test_shared SRCS shared_test.cc DEPS core)
cc_test(test_graph_utils1 SRCS graph_utils_test.cc DEPS core)
cc_test(test_arithmatic SRCS arithmatic_test.cc DEPS core)
cc_test(test_cas SRCS cas_test.cc DEPS core)
