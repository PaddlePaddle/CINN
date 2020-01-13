cc_library(cinn_gtest_main SRCS gtest_main.cc DEPS gtest)

add_subdirectory(ir)
add_subdirectory(dsl)
add_subdirectory(optim)
