set(srcs
  infershape.cc
  )

set(includes
    use_pass.h
    )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/hlir/pass/${cpp}"
    CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/hlir/pass/${header}" CACHE INTERNAL "")
endforeach()
