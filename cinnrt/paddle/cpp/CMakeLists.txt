core_gather_headers()

gather_srcs(cinnrt_src SRCS
    )

foreach(cpp ${SRCS})
  set(cinnrt_src
    "${cinnrt_src};cinnrt/paddle/cpp/${cpp}"
    CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
