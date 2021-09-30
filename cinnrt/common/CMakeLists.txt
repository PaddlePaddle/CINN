core_gather_headers()
set(core_includes "${core_includes};cinnrt/common/dtype.def" CACHE INTERNAL "")

gather_srcs(cinnrt_src SRCS
    dtype.cc
    global.cc
    target.cc
    type.cc
    shared.cc
    object.cc
    axis.cc
    string.cc
    buffer.cc
    memory.cc
    )
