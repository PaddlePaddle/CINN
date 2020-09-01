set(srcs
  var_desc.cc
  op_desc.cc
  block_desc.cc
  )

foreach(cpp ${srcs})
  set(core_src "${core_src};cinn/frontend/paddle/pb/${cpp}" CACHE INTERNAL "")
endforeach()
