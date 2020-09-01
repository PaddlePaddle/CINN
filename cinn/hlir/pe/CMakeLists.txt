set(srcs
        broadcast.cc
        elementwise.cc
        nn.cc
        reduction.cc
        schedule.cc
        transform.cc
        vision.cc
        )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/hlir/pe/${cpp}"
      CACHE INTERNAL "")
endforeach()

cc_test(test_pe_elementwise SRCS pe_elementwise_test.cc DEPS core)
cc_test(test_pe_broadcast SRCS pe_broadcast_test.cc DEPS core)
cc_test(test_pe_transform SRCS pe_transform_test.cc DEPS core)
