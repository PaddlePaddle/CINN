set(srcs
  outputs.cc
      codegen_c.cc
  )

foreach(cpp ${srcs})
  set (core_src "${core_src};cinn/backends/${cpp}" CACHE INTERNAL "")
endforeach()
