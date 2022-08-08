core_gather_headers()

gather_srcs(cinnapi_src SRCS cost_model.cc feature.cc feature_extractor.cc)

set(Python_VIRTUALENV FIRST)
find_package(PythonInterp ${PY_VERSION} REQUIRED)
find_package(PythonLibs ${PY_VERSION} REQUIRED)


cc_test(test_cost_model SRCS cost_model_test.cc cost_model.cc DEPS pybind gtest_main)
cc_test(test_feature_extractor SRCS feature_extractor_test.cc DEPS cinncore)
cc_test(test_feature SRCS feature_test.cc DEPS cinncore)

target_link_libraries(test_cost_model ${PYTHON_LIBRARIES})
