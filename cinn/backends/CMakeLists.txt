set(srcs outputs.cc
        codegen_c.cc
        codegen_c_x86.cc
        codegen_c_x86_test.cc
        codegen_cuda_dev.cc
  )

if (WITH_CUDA)
    set(srcs ${srcs} cuda_util.cc nvrtc_util.cc)
endif()

cc_library(__x86_source_fake_lib SRCS _x86_builtin_source.cc)

cc_test(test_codegen_c SRCS codegen_c_test.cc DEPS core ARGS ${global_test_args})
cc_test(test_codegen_c_x86 SRCS codegen_c_x86_test.cc DEPS core ARGS ${global_test_args})
cc_test(test_generated1 SRCS ${CMAKE_BINARY_DIR}/cinn/backends/generated_module1.cc DEPS cinn_runtime)
include_directories(${CMAKE_SOURCE_DIR}/cinn/runtime)
add_dependencies(test_generated1 test_codegen_c)

cc_test(test_codegen_cuda SRCS codegen_cuda_dev_test.cc DEPS core)

if(WITH_CUDA)
  nv_test(test_nvrtc_util SRCS nvrtc_util_test.cc DEPS core)

  cc_library(generated1_cuda SRCS ${CMAKE_BINARY_DIR}/cinn/backends/generated1.cu DEPS core)
  add_dependencies(generated1_cuda test_codegen_cuda)
endif()

foreach(cpp ${srcs})
  set(core_src
          "${core_src};cinn/backends/${cpp}"
          CACHE INTERNAL "")
endforeach()
add_subdirectory(llvm)
