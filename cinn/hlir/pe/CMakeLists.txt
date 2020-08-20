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

cc_test(test_pe_elementwise SRCS test_pe_elementwise.cc DEPS core)
cc_test(test_pe_broadcast SRCS test_pe_broadcast.cc DEPS core)
