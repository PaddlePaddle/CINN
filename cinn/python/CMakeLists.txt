cc_library(cinn_core SHARED SRCS python_api.cc
        hlir_api_wrapper.cc
        DEPS cinn_runtime core pybind)

execute_process(COMMAND python -m pybind11 --includes OUTPUT_VARIABLE pybind_includes)
string(REGEX REPLACE "\n$" "" pybind_includes "${pybind_includes}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${pybind_includes}")


add_test(NAME test_python_api
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/cinn/python:$ENV{PYTHONPATH}
  python ${CMAKE_CURRENT_SOURCE_DIR}/cinn/test_python_api.py)

SET_TARGET_PROPERTIES(cinn_core PROPERTIES PREFIX "")
