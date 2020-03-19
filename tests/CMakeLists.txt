include_directories(${CMAKE_SOURCE_DIR}/cinn/runtime)


cc_test(test01_elementwise_add_main SRCS test01_elementwise_add_main.cc DEPS core
  ARGS ${global_test_args}
  )
cc_test(test01_elementwise_add_case
  SRCS test01_elementwise_add_case.cc ${CMAKE_BINARY_DIR}/tests/test01_elementwise_add.cc
  DEPS core)
add_dependencies(test01_elementwise_add_case test01_elementwise_add_main)

cc_test(test02_matmul_main SRCS test02_matmul_main.cc DEPS core
  ARGS ${global_test_args}
  )
cc_test(test02_matmul_case SRCS test02_matmul_case.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_tile.cc
  DEPS core)
add_dependencies(test02_matmul_case  test02_matmul_main)
