add_subdirectory(framework)
add_subdirectory(pe)
add_subdirectory(op)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/hlir/${cpp}"
      CACHE INTERNAL "")
endforeach()
