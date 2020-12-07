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

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()

cc_test(test_cinn_pe_elementwise SRCS pe_elementwise_test.cc DEPS cinncore)
cc_test(test_cinn_pe_broadcast SRCS pe_broadcast_test.cc DEPS cinncore)
cc_test(test_cinn_pe_transform SRCS pe_transform_test.cc DEPS cinncore)
