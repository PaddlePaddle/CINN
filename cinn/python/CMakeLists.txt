cc_library(python_api SRCS python_api.cc
        hlir_api_wrapper.cc
        DEPS core pybind)
