set(srcs
        broadcast.cc)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/hlir/pe/${cpp}"
      CACHE INTERNAL "")
endforeach()
