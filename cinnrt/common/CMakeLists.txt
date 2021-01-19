set(srcs
        dtype.cc
        )

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinnrt/common/${cpp}"
            CACHE INTERNAL "")
endforeach()
