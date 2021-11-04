set(srcs
  tensor_map.cc
  tensor_shape.cc
  tensor_metadata.cc
  dense_host_tensor.cc
  dense_tensor_view.cc
  )

foreach(cpp ${srcs})
  set(infrt_src
    "${infrt_src};infrt/tensor/${cpp}"
    CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()

set(tensor_map_mlir "${CMAKE_SOURCE_DIR}/infrt/dialect/mlir_tests/tensor_map.mlir")
set(external_kernels_lib "${CMAKE_BINARY_DIR}/paddle/libexternal_kernels.so")
message(STATUS "tensor_map_mlir: ${tensor_map_mlir}")
message(STATUS "external_kernels_lib: ${external_kernels_lib}")
add_test(
    NAME run_and_check_tensor_map
    COMMAND sh -c "sed -e 's|/cinn/build|${CMAKE_BINARY_DIR}|' ${tensor_map_mlir} > /tmp/tensor_map.mlir && ${CMAKE_BINARY_DIR}/infrt/host_context/cinn-exec -i /tmp/tensor_map.mlir --shared_libs=${external_kernels_lib} | ${LLVM_PATH}/bin/FileCheck ${tensor_map_mlir}"
)
