set(srcs
        display_program.cc
        buffer_assign_pass.cc
        )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};${CMAKE_SOURCE_DIR}/cinn/hlir/instruction/pass/${cpp}"
    CACHE INTERNAL "")
endforeach()
