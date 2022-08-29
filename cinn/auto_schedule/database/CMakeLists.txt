proto_library(tuning_record_proto SRCS tuning_record.proto)

core_gather_headers()

gather_srcs(cinnapi_src SRCS database.cc json_database.cc)

cc_test(test_database SRCS database_test.cc DEPS cinncore)
cc_test(test_json_database SRCS json_database_test.cc DEPS cinncore)

foreach(header ${tuning_record_proto_HDRS})
  set(core_proto_includes "${core_proto_includes};${header}" CACHE INTERNAL "")
endforeach()
