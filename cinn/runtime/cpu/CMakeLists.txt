set(srcs host_intrinsics.cc)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinn/runtime/cpu/${cpp}"
            CACHE INTERNAL "")
endforeach()
