include_directories(${CMAKE_SOURCE_DIR}/cinn/runtime)
set(srcs test_utils.cc test_matmul.cc test_elementwise.cc)

cc_test(test_matmul SRCS test_matmul.cc test_utils.cc DEPS core ARGS ${global_test_args})
target_compile_options(test_matmul PRIVATE "-O3")

cc_test(test_elementwise SRCS test_elementwise.cc test_utils.cc DEPS core ARGS ${global_test_args})
target_compile_options(test_elementwise PRIVATE "-O3")
