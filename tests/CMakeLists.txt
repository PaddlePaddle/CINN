include_directories(${CMAKE_SOURCE_DIR}/cinn/runtime)

cc_test(test01_elementwise_add_main SRCS test01_elementwise_add_main.cc DEPS core)
cc_test(test01_elementwise_add_case
  SRCS test01_elementwise_add_case.cc ${CMAKE_BINARY_DIR}/tests/test01_matmul_gen.cc
  DEPS core)
