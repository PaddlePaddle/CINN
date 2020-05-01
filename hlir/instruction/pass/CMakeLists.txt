set(srcs
  display_program.cc)

foreach(cpp ${srcs})
  set(hlir_src
    "${hlir_src};${CMAKE_SOURCE_DIR}/hlir/instruction/pass/${cpp}"
    CACHE INTERNAL "")
endforeach()
