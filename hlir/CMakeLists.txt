add_subdirectory(instruction)

cc_library(hlir SRCS ${hlir_src} DEPS core)
