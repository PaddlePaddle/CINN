set(srcs runtime.cc common.cc lang.cc ir.cc poly.cc backends.cc bind.cc optim.cc pe.cc frontend.cc framework.cc utils.cc)

if (WITH_CUDA)
  message(STATUS "Compile core_api with CUDA support")
  nv_library(core_api SHARED
      SRCS ${srcs}
      DEPS cinncore_static cinn_runtime pybind)
  message("cuda_nvrtc: ${CUDA_NVRTC}")
  target_link_libraries(core_api ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES} cuda cudnn)
  if (NVTX_FOUND)
    target_link_libraries(core_api ${CUDA_NVTX_LIB})
  endif()
else()
  message(STATUS "Compile core_api without CUDA support")
  cc_library(core_api SHARED
      SRCS ${srcs}
      DEPS cinncore_static cinn_runtime pybind ${llvm_libs})
endif()

target_link_libraries(core_api ${MKLML_LIB} isl ginac)
if (USE_OPENMP STREQUAL "gnu")
  target_link_libraries(core_api ${OpenMP_CXX_LIBRARIES})
  message(STATUS "OpenMP lib: ${OpenMP_CXX_LIBRARIES}")
elseif(USE_OPENMP STREQUAL "intel")
  target_link_libraries(core_api ${MKLML_IOMP_LIB})
  message(STATUS "OpenMP lib: ${MKLML_IOMP_LIB}")
endif()

SET_TARGET_PROPERTIES(core_api PROPERTIES PREFIX "")
