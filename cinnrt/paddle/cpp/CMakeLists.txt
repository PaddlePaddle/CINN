core_gather_headers()

core_gather_srcs(SRCS
    )

foreach(cpp ${SRCS})
  set(core_src
    "${core_src};cinnrt/paddle/cpp/${cpp}"
    CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
