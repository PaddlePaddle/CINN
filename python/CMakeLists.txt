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

SET(BASIC_TEST_NAMES
    test_matmul
    test_common
    test_packed_func
    test_pe_elementwise
    test_pe_reduction
    test_pe_transform
    test_op_broadcast
    test_op_transform
)

foreach(basic_test_name ${BASIC_TEST_NAMES})
    ADD_TEST(NAME ${basic_test_name}
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/${basic_test_name}.py WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach()

if (NOT ${WITH_CUDA})
    ADD_TEST(NAME test_op_nn
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_nn.py WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

ADD_TEST(NAME test_cinn_frontend
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_frontend.py
    ${CMAKE_BINARY_DIR}/thirds/naive_mul_model
    ${CMAKE_BINARY_DIR}/thirds/multi_fc_model
    "${WITH_CUDA}" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

ADD_TEST(NAME test_netbuilder
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_netbuilder.py "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

ADD_TEST(NAME test_cinnbuilder
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cinnbuilder.py "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

ADD_TEST(NAME test_cinn_ops_check
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ops.py "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

ADD_TEST(NAME test_cinn_op_benchmark
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_benchmark.py "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

ADD_TEST(NAME test_cinn_fake_resnet
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_resnet.py "${CMAKE_BINARY_DIR}/thirds/resnet_model" "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ToDo@haoze: Temporaryly close test of mobilenetV1, mobilenteV2, resnet18 and squeezenet.
# Will reopen them in the future.

#[[
ADD_TEST(NAME test_cinn_real_resnet18
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_resnet18.py "${CMAKE_BINARY_DIR}/thirds/ResNet18" "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
]]

#[[
ADD_TEST(NAME test_cinn_real_mobilenetV2
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mobilenetv2.py "${CMAKE_BINARY_DIR}/thirds/MobileNetV2" "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
]]

ADD_TEST(NAME test_cinn_real_efficientnet
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_efficientnet.py "${CMAKE_BINARY_DIR}/thirds/EfficientNet" "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

#[[
ADD_TEST(NAME test_cinn_real_mobilenetV1
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mobilenetv1.py "${CMAKE_BINARY_DIR}/thirds/MobilenetV1" "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
]]

ADD_TEST(NAME test_cinn_real_resnet50
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_resnet50.py "${CMAKE_BINARY_DIR}/thirds/ResNet50" "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

#[[
ADD_TEST(NAME test_cinn_real_squeezenet
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_squeezenet.py "${CMAKE_BINARY_DIR}/thirds/SqueezeNet" "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
]]

ADD_TEST(NAME test_cinn_real_facedet
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_facedet.py "${CMAKE_BINARY_DIR}/thirds/FaceDet" "${WITH_CUDA}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

ADD_TEST(NAME test_op_fusion
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_op_fusion.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

file(GLOB CINN_OP_TEST RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "tests/ops/test_*.py")
set(EXCLUDE_OP test_mul_op)

ADD_TEST(NAME test_mul_op
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/ops/test_mul_op.py "${WITH_CUDNN}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

foreach(op_test_name ${EXCLUDE_OP})
    list(REMOVE_ITEM CINN_OP_TEST tests/ops/${op_test_name}.py)
endforeach()

foreach(op_test_name ${CINN_OP_TEST})
    string(REGEX REPLACE ".py" "" op_test_name ${op_test_name})
    ADD_TEST(NAME ${op_test_name}
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${op_test_name}.py
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach()
