set(srcs
        dialect.cc
        types.cc
        basic_kernels.cc
        cinn_base.cc
        init_cinn_dialects.cc
        )

mlir_tablegen_on(ops)
mlir_tablegen_on(basic_kernels)
mlir_tablegen_on(cinn_base)

add_executable(cinn_opt opt.cc)
target_link_libraries(cinn_opt core ${mlir_libs})

add_test(test_mlir_opt_on_basic ${CMAKE_BINARY_DIR}/cinn/dialect/cinn_opt
        ${CMAKE_SOURCE_DIR}/cinn/dialect/mlir_tests/basic.mlir)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/dialect/${cpp}"
      CACHE INTERNAL "")
endforeach()
