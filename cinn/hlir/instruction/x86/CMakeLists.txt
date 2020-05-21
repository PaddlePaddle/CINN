set(srcs
        mkl_math.cc
        mkl_math_registors.cc
        )

cc_test(test_mkl_math SRCS mkl_math_test.cc DEPS core)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};${CMAKE_SOURCE_DIR}/cinn/hlir/instruction/x86/${cpp}"
            CACHE INTERNAL "")
endforeach()
