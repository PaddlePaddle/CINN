proto_library(framework_proto SRCS framework.proto)

add_subdirectory(cpp)
add_subdirectory(pb)

set(srcs
  model_parser.cc
  compatible_pb.cc
  )

set(includes
  compatible_pb.h
  model_parser.h
  )

cc_test(test_model_parser SRCS model_parser_test.cc DEPS cinncore
  ARGS --model_dir=${THIRD_PARTY_PATH}/model/lite_naive_model)

foreach(cpp ${srcs})
  set(core_src "${core_src};cinn/frontend/paddle/${cpp}" CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/frontend/paddle/${header}" CACHE INTERNAL "")
endforeach()
