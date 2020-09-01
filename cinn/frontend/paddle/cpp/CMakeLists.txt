set(srcs
  op_desc.cc
  block_desc.cc
  )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/frontend/paddle/cpp/${cpp}"
    CACHE INTERNAL "")
endforeach()
