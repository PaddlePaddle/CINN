core_gather_headers()

gather_srcs(cinnapi_src SRCS cost_model.cc)

set(Python_VIRTUALENV FIRST)
find_package(PythonInterp ${PY_VERSION} REQUIRED)
find_package(PythonLibs ${PY_VERSION} REQUIRED)



#cc_test(test_cost_model SRCS cost_model_test.cc cost_model.cc DEPS pybind NOT_REMOVE_GFLAG)

#target_link_libraries(test_cost_model ${PYTHON_LIBRARIES})

add_executable(test_cost_model cost_model_test.cc cost_model.cc)
add_dependencies(test_cost_model pybind gtest_main gtest extern_gtest)
get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
target_link_libraries(test_cost_model ${PYTHON_LIBRARIES} ${os_dependency_modules} pybind gtest_main gtest)
add_test(NAME test_cost_model
         COMMAND test_cost_model
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
