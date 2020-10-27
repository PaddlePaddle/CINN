cc_library(cinn_gtest_main SRCS gtest_main.cc DEPS gtest)

add_subdirectory(common)
add_subdirectory(utils)
add_subdirectory(poly)
add_subdirectory(runtime)
add_subdirectory(ir)
add_subdirectory(backends)
add_subdirectory(lang)
add_subdirectory(optim)
add_subdirectory(dialect)
add_subdirectory(hlir)
add_subdirectory(pybind)
add_subdirectory(frontend)
add_subdirectory(host_context)
add_subdirectory(kernel)


# Download a model
download_and_uncompress("${DOWNLOAD_MODEL_DIR}" "${PADDLE_RESOURCE_URL}" "lite_naive_model.tar.gz")
