set(srcs
  nn.cc
  )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/hlir/op/${cpp}"
    CACHE INTERNAL "")
endforeach()
