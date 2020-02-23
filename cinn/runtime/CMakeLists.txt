cc_library(runtime SRCS
        intrinsic.cc
        buffer.cc
        cinn_runtime.cc
        DEPS common ir)
