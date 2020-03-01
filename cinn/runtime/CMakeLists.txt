set(srcs intrinsic.cc)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/runtime/${cpp}"
      CACHE INTERNAL "")
endforeach()

cc_library(cinn_runtime SRCS cinn_runtime.cc buffer.cc cinn_x86_device_impl.cc)
