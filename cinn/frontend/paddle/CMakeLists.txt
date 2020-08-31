set(srcs
        op_desc.cc
        block_desc.cc
        )

proto_library(framework_proto SRCS framework.proto)

foreach(cpp ${srcs})
    set(core_src "${core_src};cinn/frontend/paddle/${cpp}" CACHE INTERNAL "")
endforeach()
