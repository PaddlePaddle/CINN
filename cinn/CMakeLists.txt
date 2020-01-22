cc_library(cinn_gtest_main SRCS gtest_main.cc DEPS gtest)

add_subdirectory(common)
add_subdirectory(utils)
add_subdirectory(ir)
add_subdirectory(lang)
add_subdirectory(optim)
