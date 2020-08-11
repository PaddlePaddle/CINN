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
