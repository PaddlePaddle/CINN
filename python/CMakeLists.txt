FILE(GLOB_RECURSE CINN_PY_FILES ${PROJECT_SOURCE_DIR}/cinn/python/cinn/*.py)
SET(CINN_PYTHON_DIR ${PROJECT_SOURCE_DIR}/python/cinn)
SET(CINN_PYTHON_TEST_DIR ${PROJECT_SOURCE_DIR}/python/tests)
SET(CINN_CORE_API ${CMAKE_BINARY_DIR}/python/cinn/core_api.so)
ADD_CUSTOM_COMMAND(OUTPUT ${CINN_CORE_API} POST_BUILD
    COMMAND
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/cinn/pybind/core_api.so ${CINN_CORE_API}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CINN_PYTHON_DIR} ${CMAKE_BINARY_DIR}/python/cinn
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CINN_PYTHON_TEST_DIR} ${CMAKE_BINARY_DIR}/python/tests
    DEPENDS core_api ${CINN_PY_FILES})

ADD_CUSTOM_TARGET(COPY_CINN_CORE_API ALL
    DEPENDS ${CINN_CORE_API} ${CINN_PY_FILES})

ADD_TEST(NAME test_cinn_python_api
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_matmul.py
)

ADD_TEST(NAME test_cinn_common
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
  python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_common.py
  )

ADD_TEST(NAME test_cinn_packed_func
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packed_func.py
)

ADD_TEST(NAME test_cinn_pe_elementwise
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pe_elementwise.py
)

ADD_TEST(NAME test_cinn_pe_reduction
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pe_reduction.py
)

ADD_TEST(NAME test_cinn_pe_transform
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pe_transform.py
        )

ADD_TEST(NAME test_cinn_op_nn
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_nn.py
)

ADD_TEST(NAME test_cinn_op_broadcast
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_broadcast.py
)

ADD_TEST(NAME test_cinn_frontend
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_frontend.py
)
