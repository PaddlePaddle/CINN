set(srcs
        mkl_math.cc
        math_registors.cc
        cpu_intrisics.cc
        )
cc_library(cpu_math SHARED SRCS mkl_math.cc DEPS mklml)

cc_test(test_mkl_math SRCS mkl_math_test.cc DEPS core)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};${CMAKE_SOURCE_DIR}/cinn/hlir/instruction/x86/${cpp}"
            CACHE INTERNAL "")
endforeach()
