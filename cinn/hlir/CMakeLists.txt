add_subdirectory(framework)
add_subdirectory(pe)
add_subdirectory(op)
add_subdirectory(pass)
add_subdirectory(dialect)
add_subdirectory(kernels)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/hlir/${cpp}"
      CACHE INTERNAL "")
endforeach()
