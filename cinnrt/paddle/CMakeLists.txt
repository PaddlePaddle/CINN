proto_library(paddle_framework_proto SRCS framework.proto)

core_gather_headers()

core_gather_srcs(SRCS
    model_parser.cc
    scope.cc
    tensor.cc
    node.cc
    )

foreach(cpp ${SRCS})
  set(core_src
    "${core_src};cinnrt/paddle/${cpp}"
    CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()

add_subdirectory(cpp)
add_subdirectory(pb)
