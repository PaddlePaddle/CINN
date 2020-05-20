include_directories(${CMAKE_SOURCE_DIR}/cinn/runtime)

if (NOT WITH_TESTS)
  return()
endif()


cc_test(test01_elementwise_add_main SRCS test01_elementwise_add_main.cc DEPS core
  ARGS ${global_test_args}
  )
cc_test(test01_elementwise_add_case
  SRCS test01_elementwise_add_case.cc
  ${CMAKE_BINARY_DIR}/tests/test01_elementwise_add.cc
  ${CMAKE_BINARY_DIR}/tests/test01_elementwise_add_vectorize.cc
  DEPS core)
add_dependencies(test01_elementwise_add_case test01_elementwise_add_main)

cc_test(test02_matmul_main SRCS test02_matmul_main.cc DEPS core
  ARGS ${global_test_args}
  )
cc_test(test02_matmul_case SRCS test02_matmul_case.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_tile.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_split.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_block.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_vectorize.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_loop_permutation.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_array_packing.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_varient_shape.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_varient_shape_tile.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_array_packing_dynamic_shape.cc
  ${CMAKE_BINARY_DIR}/tests/test02_matmul_call.cc
  DEPS core)
target_compile_options(test02_matmul_case PRIVATE "-O3")
add_dependencies(test02_matmul_case  test02_matmul_main)

cc_test(test03_conv_main SRCS test03_convolution_main.cc DEPS core ARGS ${global_test_args})
cc_test(test03_conv_case SRCS test03_convolution_case.cc
  ${CMAKE_BINARY_DIR}/tests/test03_convolution.cc
  DEPS core)
target_compile_options(test03_conv_case PRIVATE "-O3")
add_dependencies(test03_conv_case test03_conv_main)
