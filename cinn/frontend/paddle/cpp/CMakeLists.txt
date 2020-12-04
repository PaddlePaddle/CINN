set(srcs
  var_desc.cc
  op_desc.cc
  block_desc.cc
  program_desc.cc
  )

set(includes
    block_desc.h
    desc_api.h
    op_desc.h
    program_desc.h
    var_desc.h
    )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/frontend/paddle/cpp/${cpp}"
    CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/frontend/paddle/cpp/${header}" CACHE INTERNAL "")
endforeach()
