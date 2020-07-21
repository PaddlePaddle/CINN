cc_library(core_api SHARED
    SRCS runtime.cc common.cc lang.cc ir.cc poly.cc backends.cc bind.cc
    DEPS core cinn_runtime pybind)


execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE pybind_includes)
string(REGEX REPLACE "\n$" "" pybind_includes "${pybind_includes}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${pybind_includes}")


#execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE python3_module_suffix)
SET_TARGET_PROPERTIES(core_api PROPERTIES PREFIX "")
