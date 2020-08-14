set(srcs runtime.cc common.cc lang.cc ir.cc poly.cc backends.cc bind.cc optim.cc pe.cc frontend.cc)
if (WITH_CUDA)
  message(STATUS "Compile core_api with CUDA support")
  nv_library(core_api SHARED
      SRCS ${srcs}
      DEPS core cinn_runtime pybind)
  message("cuda_nvrtc: ${CUDA_NVRTC}")
  target_link_libraries(core_api ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES} cuda)
else()
  message(STATUS "Compile core_api without CUDA support")
  cc_library(core_api SHARED
      SRCS ${srcs}
      DEPS core cinn_runtime pybind)
endif()

execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE pybind_includes)
string(REGEX REPLACE "\n$" "" pybind_includes "${pybind_includes}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${pybind_includes}")


SET_TARGET_PROPERTIES(core_api PROPERTIES PREFIX "")
