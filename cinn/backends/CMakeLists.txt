set(srcs outputs.cc codegen_c.cc)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/backends/${cpp}"
      CACHE INTERNAL "")
endforeach()


cc_test(test_codegen_c SRCS codegen_c_test.cc DEPS core)
cc_test(test_generated1 SRCS ${CMAKE_BINARY_DIR}/cinn/backends/generated_module1.cc DEPS cinn_runtime)
include_directories(${CMAKE_SOURCE_DIR}/cinn/runtime)
add_dependencies(test_generated1 test_codegen_c)
