proto_library(framework_proto SRCS framework.proto)

add_subdirectory(cpp)
add_subdirectory(pb)

set(srcs
  model_parser.cc
  compatible_pb.cc
  )

foreach(cpp ${srcs})
  set(core_src "${core_src};cinn/frontend/paddle/${cpp}" CACHE INTERNAL "")
endforeach()
