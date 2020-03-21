set(srcs string.cc functional.cc dot.cc timer.cc)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/utils/${cpp}"
      CACHE INTERNAL "")
endforeach()
