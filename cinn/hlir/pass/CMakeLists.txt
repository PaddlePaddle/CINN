set(srcs
  infershape.cc
  )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/hlir/pass/${cpp}"
    CACHE INTERNAL "")
endforeach()
