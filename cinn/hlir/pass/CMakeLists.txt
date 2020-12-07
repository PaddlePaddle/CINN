set(srcs
  infershape.cc
  )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinn/hlir/pass/${cpp}"
    CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
