set(srcs intrinsic.cc buffer.cc cinn_runtime.cc)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/runtime/${cpp}"
      CACHE INTERNAL "")
endforeach()
