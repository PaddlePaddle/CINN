set(srcs
        nn.cc
        broadcast.cc
        transform.cc
  )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/hlir/op/${cpp}"
    CACHE INTERNAL "")
endforeach()

cc_test(test_op_broadcast SRCS op_broadcast_test.cc DEPS core)
cc_test(test_op_nn SRCS op_nn_test.cc DEPS core)
