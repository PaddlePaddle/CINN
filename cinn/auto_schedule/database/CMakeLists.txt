core_gather_headers()

gather_srcs(cinnapi_src SRCS database.cc)

cc_test(test_database SRCS database_test.cc DEPS cinncore)
cc_test(test_memory_database SRCS memory_database_test.cc DEPS cinncore)
