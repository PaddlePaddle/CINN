set(srcs
        nn.cc
        broadcast.cc
        transform.cc
        elementwise.cc
  )

set(includes
    use_ops.h
    )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/hlir/op/${cpp}"
    CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/hlir/op/${header}" CACHE INTERNAL "")
endforeach()

cc_test(test_cinn_op_broadcast SRCS op_broadcast_test.cc DEPS cinncore)
cc_test(test_cinn_op_nn SRCS op_nn_test.cc DEPS cinncore)
