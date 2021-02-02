core_gather_headers()

core_gather_srcs(
  outputs.cc
  codegen_c.cc
  codegen_c_x86.cc
  codegen_c_x86_test.cc
  codegen_cuda_dev.cc
  codegen_cuda_host.cc
  extern_func_emitter.cc
  extern_func_emitter_builtin.cc
  function_prototype.cc
  extern_func_protos.cc
  extern_func_jit_register.cc
  modular.cc
  compiler.cc
)

if (WITH_CUDA)
  list(APPEND srcs cuda_util.cc nvrtc_util.cc codegen_cuda_util.cc)
endif()

cc_library(__x86_source_fake_lib SRCS _x86_builtin_source.cc)
add_subdirectory(llvm)

if (WITH_CUDA)
    nv_test(test_raw_cuda_code SRCS raw_cuda_code_test.cu DEPS cinncore)
endif()

cc_test(test_codegen_c SRCS codegen_c_test.cc DEPS cinncore ARGS ${global_test_args})
cc_test(test_codegen_c_x86 SRCS codegen_c_x86_test.cc DEPS cinncore ARGS ${global_test_args})
cc_test(test_generated1 SRCS generated_module1.cc DEPS cinn_runtime)
include_directories(${CMAKE_SOURCE_DIR}/cinn/runtime)
add_dependencies(test_generated1 test_codegen_c)


if (WITH_CUDA)
  nv_test(test_codegen_cuda_dev SRCS codegen_cuda_dev_test.cc ../common/cuda_test_helper.cc DEPS cinncore)
  nv_test(test_nvrtc_util SRCS nvrtc_util_test.cc DEPS cinncore)

  cc_library(generated1_cuda SRCS generated1.cu DEPS cinncore)
  add_dependencies(generated1_cuda test_codegen_cuda_dev)

  nv_test(test_compiler SRCS compiler_test.cc DEPS cinncore)
else()
  cc_test(test_compiler SRCS compiler_test.cc DEPS cinncore)
endif()


foreach(cpp ${srcs})
  set(core_src
          "${core_src};cinn/backends/${cpp}"
          CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
