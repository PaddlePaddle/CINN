set(srcs
        basic_kernels.cc)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinn/kernel/${cpp}"
            CACHE INTERNAL "")
endforeach()
