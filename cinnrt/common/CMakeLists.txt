set(srcs
        dtype.cc
        global.cc
        )

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinnrt/common/${cpp}"
            CACHE INTERNAL "")
endforeach()
