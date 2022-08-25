core_gather_headers()

gather_srcs(cinnapi_src SRCS cost_model.cc)

set(Python_VIRTUALENV FIRST)
find_package(PythonInterp ${PY_VERSION} REQUIRED)
find_package(PythonLibs ${PY_VERSION} REQUIRED)

if (WITH_TESTING)
  cc_test(test_cost_model SRCS cost_model_test.cc cost_model.cc DEPS pybind gtest_main)

  target_link_libraries(test_cost_model ${PYTHON_LIBRARIES})
endif()
