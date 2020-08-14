set(srcs
        host_intrinsics.cc
        mkl_math.cc
        cblas.cc
        )

cc_test(test_mkl_math SRCS mkl_math_test.cc mkl_math.cc DEPS core)
cc_test(test_host_intrinsics SRCS host_intrinsics_test.cc DEPS core)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinn/runtime/cpu/${cpp}"
            CACHE INTERNAL "")
endforeach()
